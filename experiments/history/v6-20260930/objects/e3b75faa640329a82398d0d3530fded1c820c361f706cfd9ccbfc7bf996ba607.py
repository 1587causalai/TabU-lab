"""Pure data schema objects; no model or tensor dependency."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    kind: str = "numeric"
    domain: tuple[str, ...] = ()
    column_id: int = 0

    def __post_init__(self):
        if self.kind not in ("numeric", "nominal", "ordinal"):
            raise ValueError("invalid-input: unknown feature type")
        if type(self.column_id) is not int or self.column_id < 0:
            raise ValueError("column_id must be a stable nonnegative integer")
        if self.kind != "numeric" and (
            not self.domain or len(set(self.domain)) != len(self.domain)
        ):
            raise ValueError(
                "discrete features require a nonempty unique ordered domain"
            )
