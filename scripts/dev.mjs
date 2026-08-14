import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";

const root = process.cwd();
const backendDirectory = join(root, "ai-backend");
const pythonExecutable = existsSync(
  join(backendDirectory, ".venv", "Scripts", "uvicorn.exe")
)
  ? join(backendDirectory, ".venv", "Scripts", "python.exe")
  : join(backendDirectory, "venv", "Scripts", "python.exe");

if (!existsSync(pythonExecutable)) {
  throw new Error(
    "Python virtual environment not found in ai-backend/venv or ai-backend/.venv."
  );
}

const frontend = spawn(
  process.execPath,
  [join(root, "node_modules", "next", "dist", "bin", "next"), "dev", "--turbo"],
  {
    cwd: root,
    stdio: "inherit",
  }
);
const backend = spawn(
  pythonExecutable,
  [
    "-m",
    "uvicorn",
    "app.main:app",
    "--reload",
    "--reload-dir",
    "app",
    "--port",
    "8001",
  ],
  {
    cwd: backendDirectory,
    stdio: "inherit",
  }
);

let stopping = false;
function stop(exitCode = 0) {
  if (stopping) {
    return;
  }
  stopping = true;
  frontend.kill();
  backend.kill();
  process.exit(exitCode);
}

process.on("SIGINT", () => stop());
process.on("SIGTERM", () => stop());
frontend.on("exit", (code) => stop(code ?? 0));
backend.on("exit", (code) => stop(code ?? 0));
