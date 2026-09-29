// sovereign-launcher: the double-click entry point.
//
// Its whole job is process lifecycle, so that a user never sees a console
// window or has to pick a port:
//
//   1. If Sovereign is already running, open the browser at the existing
//      instance's port and exit. Never start a second solver daemon.
//   2. Otherwise find a free loopback port, start sovereign-server on it,
//      read the port back, and open the default browser there.
//
// This is a LOCAL APPLICATION plus a LOCALHOST SERVER plus an automatically
// opened BROWSER. It is deliberately not a captive portal: no Wi-Fi control, no
// DNS interception, no traffic redirection. Nothing outside this machine is
// involved at any point.

#include <windows.h>
#include <shellapi.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {

constexpr wchar_t kWindowTitle[] = L"Sovereign";

/** Where the running instance records its port, so a second launch can find it. */
std::wstring instance_file_path() {
  wchar_t buf[MAX_PATH];
  DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", buf, MAX_PATH);
  if (n == 0 || n >= MAX_PATH) return L"sovereign-instance.txt";
  std::wstring dir(buf);
  CreateDirectoryW((dir + L"\\Sovereign").c_str(), nullptr);
  return dir + L"\\Sovereign\\instance.txt";
}

std::wstring exe_dir() {
  wchar_t buf[MAX_PATH];
  DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
  if (n == 0 || n >= MAX_PATH) return L".";
  return fs::path(buf).parent_path().wstring();
}

/** True if something is already listening on `port` as a Sovereign server. */
bool sovereign_already_serving(int port) {
  std::string req = "GET /api/health HTTP/1.1\r\n"
                    "Host: 127.0.0.1\r\n"
                    "Connection: close\r\n\r\n";
  SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
  if (s == INVALID_SOCKET) return false;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<u_short>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

  bool ok = false;
  if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
    send(s, req.data(), static_cast<int>(req.size()), 0);

    // Loop until we see the marker or the peer closes. A single recv() is NOT
    // enough: HTTP headers and body routinely arrive in separate TCP segments,
    // and the marker lives in the body. Reading once returned only the headers,
    // so the probe always reported "not running" and a second launcher started
    // a duplicate solver daemon.
    const DWORD kTimeoutMs = 2000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&kTimeoutMs), sizeof(kTimeoutMs));

    std::string acc;
    char buf[1024];
    for (int i = 0; i < 16; ++i) {  // bounded, so a hostile port cannot spin us
      const int n = recv(s, buf, sizeof(buf) - 1, 0);
      if (n <= 0) break;  // peer closed (Connection: close) or timed out
      acc.append(buf, n);
      if (acc.find("sovereign-local") != std::string::npos) {
        ok = true;
        break;
      }
    }
  }
  closesocket(s);
  return ok;
}

int read_instance_port() {
  std::ifstream in{fs::path(instance_file_path())};
  int port = 0;
  if (in >> port) return port;
  return 0;
}

void write_instance_port(int port) {
  std::ofstream out(fs::path(instance_file_path()), std::ios::trunc);
  out << port;
}

/** Open the default browser. ShellExecute with "open" is the documented way. */
void open_browser(const std::wstring& url) {
  ShellExecuteW(nullptr, L"open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

std::wstring url_for(int port) {
  return L"http://127.0.0.1:" + std::to_wstring(port);
}

[[noreturn]] void fail(const std::wstring& msg) {
  MessageBoxW(nullptr, msg.c_str(), kWindowTitle, MB_OK | MB_ICONERROR);
  std::exit(1);
}

}  // namespace

int main() {
  // Winsock is needed for the liveness probe.
  WSADATA wsa;
  WSAStartup(MAKEWORD(2, 2), &wsa);

  // 1. An instance is already running? Reuse it rather than starting a second
  //    solver daemon. This is the "do not launch unnecessarily" requirement.
  const int existing = read_instance_port();
  if (existing > 0 && sovereign_already_serving(existing)) {
    open_browser(url_for(existing));
    WSACleanup();
    return 0;
  }

  // 2. Find the server next to us, or one directory up (build layout).
  const std::wstring dir = exe_dir();
  std::error_code ec;
  std::wstring server_exe = (fs::path(dir) / L"sovereign-server.exe").wstring();
  if (!fs::exists(server_exe, ec)) {
    server_exe = (fs::path(dir) / L".." / L"solver" / L"Release" / L"sovereign-server.exe")
                     .wstring();
  }
  if (!fs::exists(server_exe, ec)) {
    fail(L"Sovereign server not found.\n\nExpected sovereign-server.exe next to "
         L"sovereign-launcher.exe, or in ..\\solver\\Release\\.");
  }

  // 3. Ask the OS for a free port by passing 0. The server prints the port it
  //    actually bound, which is the only reliable way to learn it.
  SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
  HANDLE read_pipe = nullptr, write_pipe = nullptr;
  if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
    fail(L"Could not create a pipe to read the selected port.");
  }
  SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
  si.wShowWindow = SW_HIDE;
  si.hStdOutput = write_pipe;
  si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

  PROCESS_INFORMATION pi{};

  // Port 0 => OS picks. Loopback bind is enforced inside the server itself and
  // cannot be overridden from here.
  std::wstring cmd = L"\"" + server_exe + L"\" --port 0 --print-port";

  if (!CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, TRUE,
                      CREATE_NO_WINDOW, nullptr, dir.c_str(), &si, &pi)) {
    fail(L"Could not start the Sovereign server.");
  }
  // The write end must close in the parent or the read below would never see EOF.
  CloseHandle(write_pipe);

  // 4. Read "SOVEREIGN_PORT=<n>" from stdout.
  std::string out;
  char buf[256];
  DWORD read = 0;
  const ULONGLONG deadline = GetTickCount64() + 15000;
  while (GetTickCount64() < deadline) {
    DWORD available = 0;
    if (!PeekNamedPipe(read_pipe, nullptr, 0, nullptr, &available, nullptr)) break;
    if (available > 0) {
      if (!ReadFile(read_pipe, buf, sizeof(buf)-1, &read, nullptr) || read==0) break;
      out.append(buf, read);
      if (out.find('\n') != std::string::npos) break;
    } else {
      if (WaitForSingleObject(pi.hProcess, 0) == WAIT_OBJECT_0) break;
      Sleep(25);
    }
  }
  CloseHandle(read_pipe);

  int port = 0;
  const std::string key = "SOVEREIGN_PORT=";
  const auto pos = out.find(key);
  if (pos != std::string::npos) {
    port = std::atoi(out.c_str() + pos + key.size());
  }
  if (port <= 0) {
    TerminateProcess(pi.hProcess, 1);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    fail(L"The Sovereign server did not report a port.\n\nIt may have failed to start. "
         L"Run sovereign-server.exe directly to see the error.");
  }

  // 5. Record the port and open the browser. The server keeps running detached;
  //    the user closes it from the tray/console or by ending the process.
  write_instance_port(port);
  open_browser(url_for(port));

  // Detach: do not wait, or the launcher would block until the server exits.
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);

  WSACleanup();
  return 0;
}
