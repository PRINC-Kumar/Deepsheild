/**
 * js/layout.js — Sidebar, topbar, mock-mode banner injection
 */
function buildSidebar(activePage) {
  const user = DS.getUser();
  const navItems = [
    { href: '/pages/dashboard.html', icon: '📊', label: 'Dashboard', key: 'dashboard' },
    { href: '/pages/detect.html',    icon: '🔍', label: 'Upload & Detect', key: 'detect' },
    { href: '/pages/webcam.html',    icon: '📷', label: 'Live Webcam', key: 'webcam' },
    { href: '/pages/history.html',   icon: '📋', label: 'History', key: 'history' },
    { href: '/pages/about.html',     icon: '📖', label: 'How It Works', key: 'about' },
    { href: '/pages/settings.html',  icon: '⚙️', label: 'Settings', key: 'settings' },
  ];

  const nav = navItems.map(item => `
    <a href="${item.href}" class="nav-link ${activePage === item.key ? 'active' : ''}">
      <span>${item.icon}</span> ${item.label}
    </a>
  `).join('');

  return `
    <div class="sidebar-logo">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      </svg>
      <span>DeepShield</span>
    </div>
    <nav class="sidebar-nav">
      <div class="nav-section">Navigation</div>
      ${nav}
      <div class="nav-section" style="margin-top:1rem">Account</div>
      <a href="#" onclick="DS.logout()" class="nav-link">
        <span>🚪</span> Sign Out
      </a>
    </nav>
    <div style="padding:1rem; border-top:1px solid #334155; font-size:0.75rem; color:#475569;">
      ${user ? user.email : ''}
    </div>
  `;
}

async function initLayout(activePage) {
  if (!DS.requireAuth()) return;

  // Mock banner
  try {
    const health = await fetch('/api/health').then(r => r.json());
    if (health.mock_mode) {
      const banner = document.createElement('div');
      banner.className = 'mock-banner';
      banner.textContent = '⚠️ MOCK MODE — Results are randomly generated for development. Set USE_MOCK_ML=false in .env to use your real model.';
      document.body.prepend(banner);
    }
  } catch {}

  // Sidebar
  const sidebar = document.querySelector('.sidebar');
  if (sidebar) sidebar.innerHTML = buildSidebar(activePage);

  // Topbar user info
  const userEl = document.getElementById('topbar-user');
  const user = DS.getUser();
  if (userEl && user) userEl.textContent = user.email;

  // Mobile toggle
  const toggle = document.getElementById('sidebar-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      document.querySelector('.sidebar').classList.toggle('open');
    });
  }
}

window.initLayout = initLayout;

