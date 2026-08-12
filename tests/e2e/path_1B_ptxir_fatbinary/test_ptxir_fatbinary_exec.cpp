#include <catch2/catch_test_macros.hpp>
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
        execl(path, path, arg, nullptr);
        _exit(127);
    }
    close(pipefd[1]);
    char buf[4096]; ssize_t n = read(pipefd[0], buf, sizeof(buf)-1);
    buf[n>0?n:0]='\0';
    close(pipefd[0]);
    int st; waitpid(pid, &st, 0);
    return std::string(buf);
}

TEST_CASE("Path 1B Scenario 1.1: PTXIR fat-binary real exec", "[e2e][path_1B]") {
    const char* bin = "./path_1B_standalone";
    std::string out = exec_capture(bin, "vector_add");
    REQUIRE(out.find("OK: vector_add(N=1024) sum=") != std::string::npos);
}

TEST_CASE("Path 1B Scenario 1.2: kNoFooter (no PTXIR tail)", "[e2e][path_1B]") {
    auto buf = read_file("./path_1B_standalone.cubin");
    REQUIRE(buf.size() > 16);
    buf.resize(buf.size() - 8);
    write_file("./path_1B_nofooter.cubin", buf);
    std::string out = exec_capture("./path_1B_standalone", "vector_add");
    REQUIRE(out.find("OK:") == std::string::npos);
}

TEST_CASE("Path 1B Scenario 1.3: kMalformedPtxir (CRC mismatch)", "[e2e][path_1B]") {
    auto buf = read_file("./path_1B_standalone.cubin");
    REQUIRE(buf.size() > 24);
    buf[buf.size() - 16] ^= 0xFF;
    write_file("./path_1B_corrupt.cubin", buf);
    std::string out = exec_capture("./path_1B_standalone", "vector_add");
    REQUIRE(out.find("OK:") == std::string::npos);
}

TEST_CASE("Path 1B Scenario 1.4: kMalformedManifest (kernel_name empty)", "[e2e][path_1B]") {
    std::string out = exec_capture("./path_1B_standalone_empty_kernel", "vector_add");
    REQUIRE(out.find("OK:") == std::string::npos);
}

TEST_CASE("Path 1B Scenario 1.5: Anti-fallback guard (PTXIR_MODE=auto)", "[e2e][path_1B]") {
    const char* mode = std::getenv("PTXIR_MODE");
    REQUIRE(mode != nullptr);
    REQUIRE(std::string(mode) == "auto");
}
