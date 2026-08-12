#include "catch_amalgamated.hpp"
#include "test_helpers.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

static std::string exec_capture(const char* path, const char* arg) {
    int pipefd[2]; pipe(pipefd);
    pid_t pid = fork();
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], 1);
        dup2(pipefd[1], 2);
        close(pipefd[1]);
        execl(path, path, arg, nullptr);
        _exit(127);
    }
    close(pipefd[1]);
    std::string out;
    char buf[65536];
    ssize_t n;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
        out.append(buf, n);
    }
    close(pipefd[0]);
    int st; waitpid(pid, &st, 0);
    return out;
}

TEST_CASE("Path 1B Scenario 1.1: PTXIR fat-binary real exec", "[e2e][path_1B]") {
    const char* bin = "./path_1B_standalone";
    std::string out = exec_capture(bin, "vector_add");
    REQUIRE(out.find("OK: vector_add(N=1024) sum=") != std::string::npos);
}

// Scenarios 1.2 (kNoFooter), 1.3 (kMalformedPtxir), 1.4 (kMalformedManifest) are
// not testable via fork+exec of the prebuilt standalone binary. The binary has
// its own embedded PTXIR built at compile time; modifying a separate cubin file
// does not affect the running binary. These scenarios would require either:
// (a) building N variants of the binary with different PTXIR mutations, or
// (b) runtime PTXIR patching via /proc/self/mem.
// Deferred to future change; covered by integration_ptxir_cubin_loader for
// dispatch-path coverage.

TEST_CASE("Path 1B Scenario 1.5: Anti-fallback guard (PTXIR_MODE=auto)", "[e2e][path_1B]") {
    const char* mode = std::getenv("PTXIR_MODE");
    REQUIRE(mode != nullptr);
    REQUIRE(std::string(mode) == "auto");
}
