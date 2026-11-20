const { app, BrowserWindow, shell, ipcMain, Tray, Menu } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const { autoUpdater } = require("electron-updater");

const NEXT_PORT = 3000;
const BACKEND_PORT = 8000;

let mainWindow = null;
let tray = null;
let backendProc = null;
let staticServer = null;

// ─── System Tray ───────────────────────────────────────────────────────────────
function createTray() {
  const iconPath = path.join(__dirname, "..", "build", "icon.png");
  
  tray = new Tray(iconPath);
  const contextMenu = Menu.buildFromTemplate([
    { label: "Show App", click: () => {
      if (mainWindow) {
        mainWindow.show();
        mainWindow.focus();
      }
    }},
    { label: "Check for Updates", click: () => autoUpdater.checkForUpdatesAndNotify() },
    { type: "separator" },
    { label: "Quit", click: () => {
      if (backendProc) backendProc.kill();
      app.quit();
    }}
  ]);

  tray.setToolTip("R U Socrates");
  tray.setContextMenu(contextMenu);
  
  tray.on("double-click", () => {
    if (mainWindow) {
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

// ─── Auto Updater ─────────────────────────────────────────────────────────────
function setupAutoUpdater() {
  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;

  autoUpdater.on("checking-for-update", () => {
    if (mainWindow) {
      mainWindow.webContents.send("update-status", "Checking for updates...");
    }
  });

  autoUpdater.on("update-available", () => {
    if (mainWindow) {
      mainWindow.webContents.send("update-status", "Update available, downloading...");
    }
  });

  autoUpdater.on("update-not-available", () => {
    if (mainWindow) {
      mainWindow.webContents.send("update-status", "No updates available");
    }
  });

  autoUpdater.on("error", (err) => {
    if (mainWindow) {
      mainWindow.webContents.send("update-status", `Update error: ${err.message}`);
    }
  });

  autoUpdater.on("download-progress", (progressObj) => {
    const message = `Download speed: ${progressObj.bytesPerSecond} - Downloaded ${progressObj.percent}% (${progressObj.transferred}/${progressObj.total})`;
    if (mainWindow) {
      mainWindow.webContents.send("update-status", message);
    }
  });

  autoUpdater.on("update-downloaded", () => {
    if (mainWindow) {
      mainWindow.webContents.send("update-status", "Update downloaded, restart to install");
    }
  });
}

// ─── Serve the static export ─────────────────────────────────────────────────
// The web app is built with `output: "export"` into apps/web/out, so the
// packaged app needs no Node server and no node_modules at runtime.
//
// It is served over loopback HTTP rather than loaded via `loadFile` on purpose:
// a file:// page sends `Origin: null`, which the FastAPI CORS policy rejects.
// Loopback keeps a real http:// origin.
const STATIC_ROOT = path.join(__dirname, "..", "out");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
};

function startStaticServer() {
  return new Promise((resolve) => {
    const http = require("http");
    const fs = require("fs");

    if (!fs.existsSync(STATIC_ROOT)) {
      // Development only: fall back to the Next.js dev server. A packaged app
      // without a build is a broken install, so fail loudly there.
      if (app.isPackaged) {
        console.error(`[Static] ${STATIC_ROOT} missing from the installed app.`);
        resolve();
        return;
      }
      console.warn("[Static] out/ not found — falling back to `next dev`.");
      const npm = process.platform === "win32" ? "npm.cmd" : "npm";
      const dev = spawn(npm, ["run", "dev", "--", "-p", String(NEXT_PORT)], {
        cwd: path.join(__dirname, ".."),
        stdio: ["ignore", "pipe", "pipe"],
      });
      dev.stdout.on("data", (d) => process.stdout.write(`[Next] ${d}`));
      dev.stderr.on("data", (d) => process.stderr.write(`[Next ERR] ${d}`));
      setTimeout(resolve, 30000);
      return;
    }

    const server = http.createServer((req, res) => {
      const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
      let filePath = path.normalize(path.join(STATIC_ROOT, urlPath));

      // Refuse anything that escapes the export directory.
      if (!filePath.startsWith(STATIC_ROOT)) {
        res.writeHead(403).end();
        return;
      }
      if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        filePath = path.join(STATIC_ROOT, "index.html");
      }

      res.writeHead(200, {
        "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream",
      });
      fs.createReadStream(filePath).pipe(res);
    });

    staticServer = server;
    server.on("error", (err) => {
      console.error("[Static] Failed to start:", err.message);
      resolve();
    });
    server.listen(NEXT_PORT, "127.0.0.1", () => resolve());
  });
}

// ─── Start FastAPI backend ────────────────────────────────────────────────────
function startBackend() {
  const python = process.platform === "win32" ? "python" : "python3";
  backendProc = spawn(python, [
    "-m", "uvicorn",
    "services.api.main:app",
    "--host", "127.0.0.1",
    "--port", String(BACKEND_PORT),
  ], {
    cwd: path.join(__dirname, "..", "..", ".."),
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  backendProc.stdout.on("data", (d) => {
    const line = d.toString().trim();
    process.stdout.write(`[Backend] ${line}\n`);
  });

  backendProc.stderr.on("data", (d) => {
    process.stderr.write(`[Backend ERR] ${d.toString().trim()}\n`);
  });

  backendProc.on("error", (err) => {
    console.error("[Backend] Failed to start:", err.message);
  });
}

// ─── Browser window ───────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: "R U Socrates",
    backgroundColor: "#09090b",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.loadURL(`http://127.0.0.1:${NEXT_PORT}`);

  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.on("closed", () => { mainWindow = null; });
  
  mainWindow.on("minimize", (e) => {
    e.preventDefault();
    mainWindow.hide();
  });
}

// ─── App lifecycle ─────────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  console.log("[App] Starting R U Socrates…");

  createTray();
  setupAutoUpdater();
  
  startBackend();
  await startStaticServer();

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });

  autoUpdater.checkForUpdatesAndNotify();
});

app.on("window-all-closed", () => {
  if (backendProc) backendProc.kill();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (staticServer) staticServer.close();
  if (backendProc) backendProc.kill();
});

ipcMain.on("minimize", () => mainWindow?.hide());
ipcMain.on("maximize", () => {
  if (mainWindow?.isMaximized()) mainWindow.unmaximize();
  else mainWindow?.maximize();
});
ipcMain.on("close", () => mainWindow?.close());
ipcMain.on("check-for-updates", () => autoUpdater.checkForUpdatesAndNotify());
