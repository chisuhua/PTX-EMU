#include "ptxsim/simt_stack.h"
#include <iostream>
#include <sstream>

namespace ptxsim {

bool SIMTStackEntry::is_converged(const std::vector<ThreadState>& threads) const {
    for (size_t i = 0; i < threads.size() && i < 32; i++) {
        if (return_mask & (1u << i)) {
            if ((int)threads[i].pc != reconvergence_pc) {
                return false;
            }
        }
    }
    return true;
}

std::string SIMTStackEntry::toString() const {
    std::ostringstream oss;
    oss << "SIMTStackEntry{branch_pc=" << branch_pc 
        << ", reconvergence_pc=" << reconvergence_pc
        << ", active_mask=0x" << std::hex << active_mask
        << ", return_pc=" << std::dec << return_pc << "}";
    return oss.str();
}

void SIMTStack::push(const SIMTStackEntry& entry) {
    entries_.push_back(entry);
}

SIMTStackEntry SIMTStack::pop() {
    if (entries_.empty()) {
        throw std::runtime_error("SIMTStack is empty");
    }
    SIMTStackEntry top = entries_.back();
    entries_.pop_back();
    return top;
}

SIMTStackEntry& SIMTStack::top() {
    if (entries_.empty()) {
        throw std::runtime_error("SIMTStack is empty");
    }
    return entries_.back();
}

const SIMTStackEntry& SIMTStack::top() const {
    if (entries_.empty()) {
        throw std::runtime_error("SIMTStack is empty");
    }
    return entries_.back();
}

bool SIMTStack::empty() const {
    return entries_.empty();
}

size_t SIMTStack::depth() const {
    return entries_.size();
}

void SIMTStack::clear() {
    entries_.clear();
}

bool SIMTStack::check_reconvergence(const std::vector<ThreadState>& threads) {
    if (entries_.empty()) {
        return true;
    }
    
    SIMTStackEntry& top = entries_.back();
    
    if (top.is_converged(threads)) {
        entries_.pop_back();
        return true;
    }
    
    return false;
}

void SIMTStack::print() const {
    std::cout << "SIMT Stack (depth=" << entries_.size() << "):\n";
    for (size_t i = 0; i < entries_.size(); i++) {
        std::cout << "  [" << i << "] " << entries_[i].toString() << "\n";
    }
}

}
