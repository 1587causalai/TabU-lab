(() => {
  const button = document.querySelector('.menu-button');
  const links = document.querySelector('#nav-links');
  if (!button || !links) return;

  const close = () => {
    links.classList.remove('open');
    button.setAttribute('aria-expanded', 'false');
  };

  button.addEventListener('click', () => {
    const open = !links.classList.contains('open');
    links.classList.toggle('open', open);
    button.setAttribute('aria-expanded', String(open));
  });

  links.querySelectorAll('a').forEach((link) => link.addEventListener('click', close));
  window.addEventListener('resize', () => {
    if (window.innerWidth > 760) close();
  });
})();
