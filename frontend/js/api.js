/**
 * js/api.js — Centralised API client with auth token handling
 */
const API_BASE = '';  // same origin

function getToken() { return localStorage.getItem('ds_token'); }
function setToken(t) { localStorage.setItem('ds_token', t); }
function clearToken() { localStorage.removeItem('ds_token'); localStorage.removeItem('ds_user'); }
function getUser() { try { return JSON.parse(localStorage.getItem('ds_user') || 'null'); } catch { return null; } }
function setUser(u) { localStorage.setItem('ds_user', JSON.stringify(u)); }

async function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = { ...options.headers };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  if (!(options.body instanceof FormData) && options.body && typeof options.body === 'object') {
    headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(options.body);
  }
  const res = await fetch(API_BASE + path, { ...options, headers });
  if (res.status === 401) {
    clearToken();
    if (!window.location.pathname.includes('login') && !window.location.pathname.includes('signup') && !window.location.pathname.endsWith('/') && window.location.pathname !== '/index.html') {
      window.location.href = '/pages/login.html';
    }
    throw new Error('Session expired. Please log in again.');
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

function requireAuth() {
  if (!getToken()) {
    window.location.href = '/pages/login.html';
    return false;
  }
  return true;
}

function logout() {
  clearToken();
  window.location.href = '/pages/login.html';
}

window.DS = { apiFetch, getToken, setToken, clearToken, getUser, setUser, logout, requireAuth };

