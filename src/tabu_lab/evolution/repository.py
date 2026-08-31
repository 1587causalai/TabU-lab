"""Manifest loading, source binding, graph compilation, and snapshot resolution."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter, ValidationError

from tabu_lab.contracts import canonical_hash, canonical_json

from .models import (
    CompatibilityDisposition,
    CompatibilityEdge,
    ComponentGraphNode,
    ComponentNode,
    EvolutionNode,
    EvolutionNodeBase,
    EvolutionNodeKind,
    ManifestLock,
    ModelContractNode,
    NodeRef,
    ProgramLane,
    ProgramSnapshot,
    ResolvedNodeRef,
    ResolvedProgramSnapshot,
    SamplingPolicyNode,
    SourceBinding,
    StateProjectionNode,
    WorldMixtureNode,
)

_NODE_ADAPTER = TypeAdapter(EvolutionNode)


class EvolutionManifestError(ValueError):
    """A manifest repository is incomplete, ambiguous, or identity-invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvolutionManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvolutionManifestError(f"manifest root must be a mapping: {path}")
    return dict(payload)


def _load_model(path: Path, model_type: type[Any]) -> Any:
    try:
        return model_type.model_validate(_read_mapping(path))
    except ValidationError as exc:
        raise EvolutionManifestError(f"invalid manifest {path}: {exc}") from exc


def _load_node(path: Path) -> EvolutionNode:
    try:
        return _NODE_ADAPTER.validate_python(_read_mapping(path))
    except ValidationError as exc:
        raise EvolutionManifestError(f"invalid evolution node {path}: {exc}") from exc


def _qualname_object(symbol_ref: str) -> Any:
    module_name, separator, qualname = symbol_ref.partition(":")
    if not separator or not module_name or not qualname or "<locals>" in qualname:
        raise EvolutionManifestError(f"invalid Python symbol ref: {symbol_ref!r}")
    try:
        value: Any = importlib.import_module(module_name)
        for name in qualname.split("."):
            value = getattr(value, name)
    except (ImportError, AttributeError) as exc:
        raise EvolutionManifestError(f"cannot resolve Python symbol {symbol_ref!r}") from exc
    return value


def source_binding_hash(repository_root: Path, binding: SourceBinding) -> str:
    source = (repository_root / binding.source_path).resolve()
    try:
        source.relative_to(repository_root)
    except ValueError as exc:
        raise EvolutionManifestError("source binding escapes repository root") from exc
    if not source.is_file():
        raise EvolutionManifestError(f"bound source does not exist: {binding.source_path}")
    if binding.hash_mode == "file":
        return _sha256_bytes(source.read_bytes())

    assert binding.symbol_ref is not None
    value = _qualname_object(binding.symbol_ref)
    try:
        observed_source = Path(inspect.getsourcefile(value) or "").resolve()
        observed_source.relative_to(repository_root)
        if observed_source != source:
            raise EvolutionManifestError(
                f"symbol {binding.symbol_ref!r} is not defined by {binding.source_path}"
            )
        text = inspect.getsource(value)
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, EvolutionManifestError):
            raise
        raise EvolutionManifestError(
            f"cannot inspect bound Python symbol {binding.symbol_ref!r}"
        ) from exc
    return _sha256_bytes(text.encode("utf-8"))


def verify_source_binding(repository_root: Path, binding: SourceBinding) -> None:
    observed = source_binding_hash(repository_root, binding)
    if observed != binding.sha256:
        raise EvolutionManifestError(
            f"source identity mismatch for {binding.source_path}: "
            f"expected {binding.sha256}, observed {observed}"
        )


def _iter_yaml(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise EvolutionManifestError(f"missing evolution manifest directory: {directory}")
    paths = tuple(
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in {".yaml", ".yml"}
    )
    if not paths:
        raise EvolutionManifestError(f"no manifests found in {directory}")
    return paths


def _unique_by_ref(values: Iterable[Any], *, label: str) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for value in values:
        if value.ref in resolved:
            raise EvolutionManifestError(f"duplicate {label} ref: {value.ref}")
        resolved[value.ref] = value
    return dict(sorted(resolved.items()))


class EvolutionRepository:
    """Validated in-memory view of canonical evolution manifests."""

    def __init__(
        self,
        *,
        root: Path,
        nodes: Mapping[str, EvolutionNode],
        edges: Mapping[str, CompatibilityEdge],
        programs: Mapping[str, ProgramSnapshot],
        lock: ManifestLock,
    ) -> None:
        self.root = root
        self.nodes = dict(nodes)
        self.edges = dict(edges)
        self.programs = dict(programs)
        self.lock = lock

    @property
    def spec_root(self) -> Path:
        return self.root / "specs" / "evolution"

    @property
    def repository_hash(self) -> str:
        return self.lock.lock_hash

    @classmethod
    def load(
        cls,
        repository: str | Path,
        *,
        verify_sources: bool = True,
        require_lock: bool = True,
        ignore_lock: bool = False,
    ) -> EvolutionRepository:
        root = Path(repository).resolve()
        spec_root = root / "specs" / "evolution"
        nodes = _unique_by_ref(
            (_load_node(path) for path in _iter_yaml(spec_root / "nodes")),
            label="node",
        )
        edges = _unique_by_ref(
            (
                _load_model(path, CompatibilityEdge)
                for path in _iter_yaml(spec_root / "edges")
            ),
            label="edge",
        )
        programs = _unique_by_ref(
            (_load_model(path, ProgramSnapshot) for path in _iter_yaml(spec_root / "programs")),
            label="program",
        )
        lock_path = spec_root / "manifest-lock.json"
        if lock_path.is_file() and not ignore_lock:
            try:
                lock = ManifestLock.model_validate(
                    json.loads(lock_path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError, ValidationError) as exc:
                raise EvolutionManifestError(f"invalid evolution manifest lock: {exc}") from exc
        elif require_lock and not ignore_lock:
            raise EvolutionManifestError("specs/evolution/manifest-lock.json is missing")
        else:
            lock = cls.build_lock(nodes=nodes, edges=edges, programs=programs)
        instance = cls(root=root, nodes=nodes, edges=edges, programs=programs, lock=lock)
        instance._validate_lock()
        instance._validate_graph()
        if verify_sources:
            instance._verify_sources()
        return instance

    @staticmethod
    def build_lock(
        *,
        nodes: Mapping[str, EvolutionNode],
        edges: Mapping[str, CompatibilityEdge],
        programs: Mapping[str, ProgramSnapshot],
    ) -> ManifestLock:
        return ManifestLock(
            nodes={ref: node.node_hash for ref, node in sorted(nodes.items())},
            edges={ref: edge.edge_hash for ref, edge in sorted(edges.items())},
            programs={ref: program.program_hash for ref, program in sorted(programs.items())},
        )

    def rendered_lock(self) -> str:
        return canonical_json(self.build_lock(
            nodes=self.nodes,
            edges=self.edges,
            programs=self.programs,
        ).model_dump(mode="python")) + "\n"

    def _validate_lock(self) -> None:
        expected = self.build_lock(nodes=self.nodes, edges=self.edges, programs=self.programs)
        for label in ("nodes", "edges", "programs"):
            observed_values = getattr(self.lock, label)
            expected_values = getattr(expected, label)
            if set(observed_values) != set(expected_values):
                missing = sorted(set(observed_values) - set(expected_values))
                added = sorted(set(expected_values) - set(observed_values))
                raise EvolutionManifestError(
                    f"manifest lock {label} set differs; missing={missing}, unlocked={added}"
                )
            changed = [
                ref
                for ref in sorted(expected_values)
                if observed_values[ref] != expected_values[ref]
            ]
            if changed:
                raise EvolutionManifestError(
                    f"immutable {label} changed without a new version: {changed}"
                )

    def _require_node(
        self,
        reference: NodeRef,
        kind: EvolutionNodeKind | None = None,
    ) -> EvolutionNode:
        try:
            node = self.nodes[reference.ref]
        except KeyError as exc:
            raise EvolutionManifestError(f"dangling evolution node ref: {reference.ref}") from exc
        if kind is not None and node.kind is not kind:
            raise EvolutionManifestError(
                f"{reference.ref} has kind {node.kind.value}, expected {kind.value}"
            )
        return node

    def node(self, reference: NodeRef | str) -> EvolutionNode:
        resolved = NodeRef.parse(reference) if isinstance(reference, str) else reference
        return self._require_node(resolved)

    def program(self, reference: ProgramSnapshot | str) -> ProgramSnapshot:
        if isinstance(reference, ProgramSnapshot):
            candidate = reference
        else:
            try:
                candidate = self.programs[reference]
            except KeyError as exc:
                raise EvolutionManifestError(f"unknown program snapshot: {reference}") from exc
        locked = self.programs.get(candidate.ref)
        if locked is None or locked.program_hash != candidate.program_hash:
            raise EvolutionManifestError("program snapshot is not locked by this repository")
        return candidate

    def _verify_sources(self) -> None:
        seen: set[tuple[str, str, str | None]] = set()

        def verify(binding: SourceBinding) -> None:
            key = (binding.source_path, binding.hash_mode, binding.symbol_ref)
            if key not in seen:
                verify_source_binding(self.root, binding)
                seen.add(key)

        for node in self.nodes.values():
            if isinstance(node, ModelContractNode):
                verify(node.source)
                if node.executable:
                    contract_id, separator, contract_version = node.contract_ref.rpartition("@")
                    if not separator:
                        raise EvolutionManifestError(
                            f"invalid executable contract ref: {node.contract_ref}"
                        )
                    from tabu_lab.registry import get_model_spec, model_spec_identity_payload

                    spec = get_model_spec(contract_id, contract_version)
                    observed = canonical_hash(model_spec_identity_payload(spec))
                    if observed != node.contract_hash:
                        raise EvolutionManifestError(
                            f"ModelContract hash mismatch for {node.contract_ref}"
                        )
            elif isinstance(node, ComponentNode):
                verify(node.implementation)
            elif hasattr(node, "implementation"):
                implementation = node.implementation
                if isinstance(implementation, SourceBinding):
                    verify(implementation)
            if isinstance(node, StateProjectionNode):
                validation_path, separator, test_name = node.validation_ref.partition("::")
                candidate = (self.root / validation_path).resolve()
                try:
                    candidate.relative_to(self.root)
                except ValueError as exc:
                    raise EvolutionManifestError(
                        f"StateProjection validation escapes repository: {node.ref}"
                    ) from exc
                if (
                    not separator
                    or not candidate.is_file()
                    or f"def {test_name}(" not in candidate.read_text(encoding="utf-8")
                ):
                    raise EvolutionManifestError(
                        f"StateProjection validation target is missing: {node.ref}"
                    )
        for edge in self.edges.values():
            verify(edge.verifier)

    def _validate_graph(self) -> None:
        for node in self.nodes.values():
            for dependency in node.dependency_refs():
                self._require_node(dependency)
        for edge in self.edges.values():
            source = self._require_node(edge.source)
            target = self._require_node(edge.target)
            if edge.verified is False:
                raise EvolutionManifestError(
                    f"compatibility edge must remain absent until verified: {edge.ref}"
                )
            if edge.disposition is CompatibilityDisposition.WARM_START_AVAILABLE and (
                source.kind is not EvolutionNodeKind.COMPONENT_GRAPH
                or target.kind is not EvolutionNodeKind.COMPONENT_GRAPH
            ):
                raise EvolutionManifestError(
                    f"warm-start edge endpoints must be component graphs: {edge.ref}"
                )
            if edge.disposition in {
                CompatibilityDisposition.RESCORE,
                CompatibilityDisposition.RERUN_INFERENCE,
            } and (
                source.kind is not EvolutionNodeKind.EVALUATION_PROTOCOL
                or target.kind is not EvolutionNodeKind.EVALUATION_PROTOCOL
            ):
                raise EvolutionManifestError(
                    f"evaluation compatibility edge endpoints are invalid: {edge.ref}"
                )
        self._validate_node_dependency_cycles()
        for node in self.nodes.values():
            if isinstance(node, ComponentGraphNode):
                self._validate_component_graph(node)
            elif isinstance(node, WorldMixtureNode):
                for entry in node.entries:
                    self._require_node(entry.generator, EvolutionNodeKind.GENERATOR)
            elif isinstance(node, StateProjectionNode):
                self._require_node(node.source_model, EvolutionNodeKind.MODEL_CONTRACT)
                self._require_node(node.source_graph, EvolutionNodeKind.COMPONENT_GRAPH)
                self._require_node(node.target_model, EvolutionNodeKind.MODEL_CONTRACT)
                self._require_node(node.target_graph, EvolutionNodeKind.COMPONENT_GRAPH)
                if not node.verified:
                    raise EvolutionManifestError(
                        f"unverified state projection cannot be registered: {node.ref}"
                    )
        for program in self.programs.values():
            self._validate_program(program)

    def _validate_node_dependency_cycles(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(ref: str) -> None:
            if ref in visiting:
                raise EvolutionManifestError(f"evolution node dependency cycle at {ref}")
            if ref in visited:
                return
            visiting.add(ref)
            for dependency in self.nodes[ref].dependency_refs():
                visit(dependency.ref)
            visiting.remove(ref)
            visited.add(ref)

        for ref in sorted(self.nodes):
            visit(ref)

    @staticmethod
    def _port_map(ports: Iterable[Any]) -> dict[str, Any]:
        return {port.name: port for port in ports}

    def _validate_component_graph(self, graph: ComponentGraphNode) -> None:
        self._require_node(graph.model_contract, EvolutionNodeKind.MODEL_CONTRACT)
        instances = {instance.instance_id: instance for instance in graph.components}
        components: dict[str, ComponentNode] = {}
        for instance in graph.components:
            node = self._require_node(instance.component, EvolutionNodeKind.COMPONENT)
            assert isinstance(node, ComponentNode)
            components[instance.instance_id] = node
        external_inputs = self._port_map(graph.external_inputs)
        external_outputs = self._port_map(graph.external_outputs)
        occupied_targets: set[tuple[str, str]] = set()
        used_outputs: set[tuple[str, str]] = set()
        adjacency: dict[str, set[str]] = {instance_id: set() for instance_id in instances}
        for connection in graph.connections:
            if connection.source_instance == "$input":
                source_ports = external_inputs
            else:
                if connection.source_instance not in components:
                    raise EvolutionManifestError(
                        f"unknown source instance in {graph.ref}: {connection.source_instance}"
                    )
                source_ports = self._port_map(components[connection.source_instance].outputs)
            if connection.target_instance == "$output":
                target_ports = external_outputs
            else:
                if connection.target_instance not in components:
                    raise EvolutionManifestError(
                        f"unknown target instance in {graph.ref}: {connection.target_instance}"
                    )
                target_ports = self._port_map(components[connection.target_instance].inputs)
            if connection.source_port not in source_ports:
                raise EvolutionManifestError(
                    f"unknown source port {connection.source_port!r} in {graph.ref}"
                )
            if connection.target_port not in target_ports:
                raise EvolutionManifestError(
                    f"unknown target port {connection.target_port!r} in {graph.ref}"
                )
            source_interface = source_ports[connection.source_port].interface_id
            target_interface = target_ports[connection.target_port].interface_id
            if source_interface != target_interface:
                raise EvolutionManifestError(
                    f"typed port mismatch in {graph.ref}: "
                    f"{source_interface} != {target_interface}"
                )
            target = (connection.target_instance, connection.target_port)
            if target in occupied_targets:
                raise EvolutionManifestError(f"graph input port is connected twice: {target}")
            occupied_targets.add(target)
            used_outputs.add((connection.source_instance, connection.source_port))
            if (
                connection.source_instance in instances
                and connection.target_instance in instances
            ):
                adjacency[connection.source_instance].add(connection.target_instance)
        for instance_id, component in components.items():
            missing = [
                port.name
                for port in component.inputs
                if port.required and (instance_id, port.name) not in occupied_targets
            ]
            if missing:
                raise EvolutionManifestError(
                    f"required component ports are unbound in {graph.ref}: "
                    f"{instance_id}={missing}"
                )
            unused_required_outputs = [
                port.name
                for port in component.outputs
                if port.required and (instance_id, port.name) not in used_outputs
            ]
            if unused_required_outputs:
                raise EvolutionManifestError(
                    f"required component outputs are unconsumed in {graph.ref}: "
                    f"{instance_id}={unused_required_outputs}"
                )
        missing_inputs = [
            port.name
            for port in graph.external_inputs
            if port.required and ("$input", port.name) not in used_outputs
        ]
        if missing_inputs:
            raise EvolutionManifestError(
                f"required graph inputs are unconsumed in {graph.ref}: {missing_inputs}"
            )
        missing_outputs = [
            port.name
            for port in graph.external_outputs
            if port.required and ("$output", port.name) not in occupied_targets
        ]
        if missing_outputs:
            raise EvolutionManifestError(
                f"required graph outputs are unbound in {graph.ref}: {missing_outputs}"
            )
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(instance_id: str) -> None:
            if instance_id in visiting:
                raise EvolutionManifestError(
                    f"component graph contains a cycle in {graph.ref}: {instance_id}"
                )
            if instance_id in visited:
                return
            visiting.add(instance_id)
            for target in adjacency[instance_id]:
                visit(target)
            visiting.remove(instance_id)
            visited.add(instance_id)

        for instance_id in sorted(instances):
            visit(instance_id)

    def _validate_program(self, program: ProgramSnapshot) -> None:
        expected = {
            "model_contract": EvolutionNodeKind.MODEL_CONTRACT,
            "component_graph": EvolutionNodeKind.COMPONENT_GRAPH,
            "world_mixture": EvolutionNodeKind.WORLD_MIXTURE,
            "sampling_policy": EvolutionNodeKind.SAMPLING_POLICY,
            "objective_bundle": EvolutionNodeKind.OBJECTIVE_BUNDLE,
            "training_recipe": EvolutionNodeKind.TRAINING_RECIPE,
            "evaluation_protocol": EvolutionNodeKind.EVALUATION_PROTOCOL,
            "state_projection": EvolutionNodeKind.STATE_PROJECTION,
        }
        for slot, reference in program.slot_refs().items():
            self._require_node(reference, expected[slot])
        graph = self._require_node(program.component_graph, EvolutionNodeKind.COMPONENT_GRAPH)
        assert isinstance(graph, ComponentGraphNode)
        if graph.model_contract.ref != program.model_contract.ref:
            raise EvolutionManifestError(
                f"program {program.ref} selects a graph for another ModelContract"
            )
        if program.state_projection is not None:
            projection = self._require_node(
                program.state_projection, EvolutionNodeKind.STATE_PROJECTION
            )
            assert isinstance(projection, StateProjectionNode)
            if (
                projection.target_model.ref != program.model_contract.ref
                or projection.target_graph.ref != program.component_graph.ref
            ):
                raise EvolutionManifestError(
                    f"program {program.ref} selects a StateProjection for another target"
                )
        policy = self._require_node(
            program.sampling_policy, EvolutionNodeKind.SAMPLING_POLICY
        )
        assert isinstance(policy, SamplingPolicyNode)
        mixture = self._require_node(
            program.world_mixture, EvolutionNodeKind.WORLD_MIXTURE
        )
        assert isinstance(mixture, WorldMixtureNode)
        generator_refs = {entry.generator.ref for entry in mixture.entries}
        for segment in policy.segments:
            if set(segment.weights) != generator_refs:
                raise EvolutionManifestError(
                    f"sampling policy keys do not match program mixture: {program.ref}"
                )
        if program.lane is ProgramLane.EVIDENCE and (
            not policy.deterministic or not policy.serializable_state
        ):
            raise EvolutionManifestError(
                f"evidence program {program.ref} requires deterministic serializable policy"
            )
        if program.parent is not None and program.parent.ref not in self.programs:
            raise EvolutionManifestError(f"dangling parent program ref: {program.parent.ref}")

    def _dependency_closure(self, roots: Iterable[NodeRef]) -> tuple[EvolutionNode, ...]:
        resolved: dict[str, EvolutionNode] = {}

        def visit(reference: NodeRef) -> None:
            if reference.ref in resolved:
                return
            node = self._require_node(reference)
            resolved[reference.ref] = node
            for dependency in node.dependency_refs():
                visit(dependency)

        for root in roots:
            visit(root)
        return tuple(resolved[ref] for ref in sorted(resolved))

    @staticmethod
    def _resolved_node(node: EvolutionNodeBase) -> ResolvedNodeRef:
        return ResolvedNodeRef(
            node_id=node.node_id,
            version=node.version,
            kind=node.kind,
            node_hash=node.node_hash,
        )

    def resolve(self, program: ProgramSnapshot | str) -> ResolvedProgramSnapshot:
        source = self.program(program)
        self._validate_program(source)
        slots = {
            slot: self._resolved_node(self._require_node(reference))
            for slot, reference in sorted(source.slot_refs().items())
        }
        closure = tuple(
            self._resolved_node(node)
            for node in self._dependency_closure(source.slot_refs().values())
        )
        manifest_closure_hash = canonical_hash(
            {
                "schema_version": "tabu.manifest-closure.v1",
                "source_program_hash": source.program_hash,
                "slots": slots,
                "dependency_closure": closure,
            }
        )
        payload = {
            "schema_version": "tabu.resolved-program-snapshot.v1",
            "program_id": source.program_id,
            "version": source.version,
            "research_question": source.research_question,
            "lane": source.lane,
            "evidence_status": source.evidence_status,
            "source_program_hash": source.program_hash,
            "slots": slots,
            "dependency_closure": closure,
            "manifest_closure_hash": manifest_closure_hash,
        }
        return ResolvedProgramSnapshot(**payload, snapshot_hash=canonical_hash(payload))

    def compatibility_edges(
        self,
        source: NodeRef | ResolvedNodeRef,
        target: NodeRef | ResolvedNodeRef,
        disposition: CompatibilityDisposition | None = None,
    ) -> tuple[CompatibilityEdge, ...]:
        values = tuple(
            edge
            for edge in self.edges.values()
            if edge.source.ref == source.ref
            and edge.target.ref == target.ref
            and edge.verified
            and (disposition is None or edge.disposition is disposition)
        )
        return tuple(sorted(values, key=lambda edge: edge.ref))


def check_or_write_lock(
    repository: str | Path,
    *,
    write: bool,
) -> str:
    """Check the lock or append new refs without permitting identity rewrites."""

    root = Path(repository).resolve()
    spec_root = root / "specs" / "evolution"
    unlocked = EvolutionRepository.load(
        root,
        require_lock=False,
        verify_sources=True,
        ignore_lock=True,
    )
    expected = unlocked.rendered_lock()
    lock_path = spec_root / "manifest-lock.json"
    if not write:
        actual = lock_path.read_text(encoding="utf-8") if lock_path.is_file() else ""
        if actual != expected:
            raise EvolutionManifestError("evolution manifest lock is stale")
        return expected
    if lock_path.is_file():
        try:
            old = ManifestLock.model_validate(json.loads(lock_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise EvolutionManifestError(f"cannot extend invalid manifest lock: {exc}") from exc
        current = EvolutionRepository.build_lock(
            nodes=unlocked.nodes,
            edges=unlocked.edges,
            programs=unlocked.programs,
        )
        for label in ("nodes", "edges", "programs"):
            old_values = getattr(old, label)
            current_values = getattr(current, label)
            removed = sorted(set(old_values) - set(current_values))
            changed = sorted(
                ref
                for ref in set(old_values).intersection(current_values)
                if old_values[ref] != current_values[ref]
            )
            if removed or changed:
                raise EvolutionManifestError(
                    f"lock extension cannot remove or rewrite {label}; "
                    f"removed={removed}, changed={changed}"
                )
    lock_path.write_text(expected, encoding="utf-8")
    return expected


__all__ = [
    "EvolutionManifestError",
    "EvolutionRepository",
    "check_or_write_lock",
    "source_binding_hash",
    "verify_source_binding",
]
