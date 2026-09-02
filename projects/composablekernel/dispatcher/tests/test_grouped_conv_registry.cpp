// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for GroupedConvRegistry and GroupedConvDispatcher using assert() and std::cout

#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <atomic>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::grouped_conv_decl;

namespace {
GroupedConvKernelKey phase0_key()
{
    GroupedConvKernelKey key;
    key.dtype_in = key.dtype_wei = key.dtype_out = "fp16";
    key.layout = "nhwgc_gkyxc_nhwgk";
    key.ndim_spatial = 2;
    key.op = GroupedConvOp::Forward;
    key.arch = "gfx1100";
    return key;
}

GroupedConvProblem phase0_problem()
{
    GroupedConvProblem problem;
    problem.dtype_in = problem.dtype_wei = problem.dtype_out = "fp16";
    problem.layout = "nhwgc_gkyxc_nhwgk";
    problem.ndim_spatial = 2;
    problem.op = GroupedConvOp::Forward;
    problem.arch = "gfx1100";
    return problem;
}

GroupedConvKernelInstancePtr executable(const GroupedConvKernelKey& key, bool supported = true)
{
    return std::make_shared<GroupedConvKernelInstance>(
        key, key.to_string(), [](const GroupedConvProblem&, void*) { return 0.0f; },
        [supported](const GroupedConvProblem&) { return supported; });
}
} // namespace

void test_grouped_conv_key_and_id()
{
    std::cout << "  test_grouped_conv_key_and_id... ";
    const auto base = phase0_key();
    const GroupedConvKernelKeyHash hash;
    assert(base.kernel_id().find("cktile_gconv_v1") == 0);

    auto check_changed = [&](auto mutate) {
        auto changed = base;
        mutate(changed);
        assert(!(changed == base));
        assert(changed.kernel_id() != base.kernel_id());
        assert(hash(changed) != hash(base));
    };
    check_changed([](auto& k) { k.dtype_wei = "bf16"; });
    check_changed([](auto& k) { k.dtype_out = "fp32"; });
    check_changed([](auto& k) { k.wave_k = 2; });
    check_changed([](auto& k) { k.memory_operation = "atomic_add"; });
    check_changed([](auto& k) { k.double_smem_buffer = true; });
    check_changed([](auto& k) { k.gemm_pad_m = false; });
    check_changed([](auto& k) { k.block_per_cu = 2; });
    check_changed([](auto& k) { k.num_wave_groups = 2; });
    check_changed([](auto& k) { k.num_groups_to_merge = 2; });

    auto split_a = base;
    auto split_b = base;
    split_a.pipeline = "a|1:b";
    split_a.scheduler = "";
    split_b.pipeline = "a";
    split_b.scheduler = "1:b|0:";
    assert(split_a.kernel_id() != split_b.kernel_id());
    std::cout << "PASSED\n";
}

void test_grouped_conv_executable_registration_validation()
{
    std::cout << "  test_grouped_conv_executable_registration_validation... ";
    GroupedConvRegistry reg;
    auto key = phase0_key();
    assert(!reg.register_kernel(key, nullptr));
    assert(!reg.register_kernel(key, std::make_shared<GroupedConvKernelInstance>(
                                        key, "missing support", [](const auto&, void*) { return 0.0f; })));
    assert(!reg.register_kernel(key, std::make_shared<GroupedConvKernelInstance>(
                                        key, "missing run", GroupedConvKernelInstance::RunFn{},
                                        [](const auto&) { return true; })));
    assert(reg.register_kernel(key, executable(key)));
    std::cout << "PASSED\n";
}

void test_grouped_conv_supported_lookup()
{
    std::cout << "  test_grouped_conv_supported_lookup... ";
    GroupedConvRegistry reg;
    auto problem = phase0_problem();
    auto a = phase0_key();
    auto b = a; b.tile_m++;
    auto unsupported = a; unsupported.tile_m += 2;
    assert(reg.register_kernel(b, executable(b), Priority::High));
    assert(reg.register_kernel(a, executable(a), Priority::High));
    assert(reg.register_kernel(unsupported, executable(unsupported, false), Priority::High));
    assert(reg.find_best_supported(problem)->key().tile_m == a.tile_m);
    auto all = reg.find_all_supported(problem);
    assert(all.size() == 2);
    assert(reg.find_by_id(problem, b.kernel_id()) != nullptr);
    assert(reg.find_by_id(problem, "unknown") == nullptr);

    auto mismatch = problem;
    mismatch.dtype_out = "fp32";
    assert(reg.find_by_id(mismatch, a.kernel_id()) == nullptr);
    mismatch = problem; mismatch.layout = "other";
    assert(reg.find_by_id(mismatch, a.kernel_id()) == nullptr);
    mismatch = problem; mismatch.ndim_spatial = 3;
    assert(reg.find_by_id(mismatch, a.kernel_id()) == nullptr);
    mismatch = problem; mismatch.arch = "gfx1101";
    assert(reg.find_by_id(mismatch, a.kernel_id()) == nullptr);
    mismatch = problem; mismatch.op = GroupedConvOp::BackwardData;
    assert(reg.find_by_id(mismatch, a.kernel_id()) == nullptr);
    mismatch = problem; mismatch.padding_right[2]++;
    assert(reg.find_all_supported(mismatch).size() == 2);

    GroupedConvRegistry reversed;
    assert(reversed.register_kernel(a, executable(a), Priority::High));
    assert(reversed.register_kernel(b, executable(b), Priority::High));
    assert(reversed.find_best_supported(problem)->kernel_id() == a.kernel_id());
    auto reversed_all = reversed.find_all_supported(problem);
    assert(reversed_all.size() == 2 && reversed_all[0]->kernel_id() == all[0]->kernel_id() &&
           reversed_all[1]->kernel_id() == all[1]->kernel_id());
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_basic()
{
    std::cout << "  test_grouped_conv_registry_basic... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    reg.set_name("test_registry");
    assert(reg.get_name() == "test_registry");

    assert(reg.size() == 0);
    assert(reg.empty());

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_register_metadata_set()
{
    std::cout << "  test_grouped_conv_registry_register_metadata_set... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    set.add("fp16", "nhwc", "forward", 256, 256);

    bool ok = reg.register_metadata_set(set);
    assert(ok);
    assert(reg.size() == 2);
    assert(!reg.empty());
    assert(reg.find_all_supported(phase0_problem()).empty());

    const auto metadata_instances = reg.all_kernels();
    const auto first_key = metadata_instances[0]->key();
    const auto second_key = metadata_instances[1]->key();
    reg.clear();
    assert(reg.register_kernel(first_key, executable(first_key)));
    assert(reg.register_kernel(second_key, executable(second_key)));
    assert(!reg.register_metadata_set(set));
    assert(reg.all_kernels().front()->executable());

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_all_kernels()
{
    std::cout << "  test_grouped_conv_registry_all_kernels... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    auto key = phase0_key();
    reg.register_kernel(key, executable(key));

    auto all = reg.all_kernels();
    assert(all.size() == 1);
    assert(all[0]->name().find("grouped_conv_") != std::string::npos);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_clear()
{
    std::cout << "  test_grouped_conv_registry_clear... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    reg.register_metadata_set(set);
    assert(reg.size() == 1);

    reg.clear();
    assert(reg.size() == 0);
    assert(reg.empty());

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_thread_safe()
{
    std::cout << "  test_grouped_conv_registry_thread_safe... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    const int num_threads     = 4;
    const int sets_per_thread = 10;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for(int t = 0; t < num_threads; t++)
    {
        threads.emplace_back([t, &reg, &success_count]() {
            for(int k = 0; k < sets_per_thread; k++)
            {
                GroupedConvKernelSet set;
                set.add("fp16", "nhwc", "forward", 128 + t * 32 + k, 128);
                if(reg.register_metadata_set(set))
                {
                    success_count++;
                }
            }
        });
    }

    for(auto& th : threads)
        th.join();

    assert(reg.size() == num_threads * sets_per_thread);
    assert(success_count.load() == num_threads * sets_per_thread);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_export_json()
{
    std::cout << "  test_grouped_conv_registry_export_json... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    reg.register_metadata_set(set);

    std::string json = reg.export_json(false);
    assert(!json.empty());
    assert(json.find("\"kernels\"") != std::string::npos);
    assert(json.find("\"metadata\"") != std::string::npos);
    assert(json.find("grouped_conv_") != std::string::npos);

    std::string json_stats = reg.export_json(true);
    assert(json_stats.find("\"statistics\"") != std::string::npos);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_filter()
{
    std::cout << "  test_grouped_conv_registry_filter... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    set.add("fp16", "nhwc", "forward", 256, 256);
    set.add("bf16", "nhwc", "forward", 128, 128);
    reg.register_metadata_set(set);

    auto fp16_only =
        reg.filter([](const GroupedConvKernelInstance& k) { return k.key().dtype_in == "fp16"; });
    assert(fp16_only.size() == 2);

    auto large_tile = reg.filter([](const GroupedConvKernelInstance& k) {
        return k.key().tile_m >= 256 || k.key().tile_n >= 256;
    });
    assert(large_tile.size() >= 1);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_dispatcher_basic()
{
    std::cout << "  test_grouped_conv_dispatcher_basic... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    auto key = phase0_key();
    reg.register_kernel(key, executable(key));

    GroupedConvDispatcher dispatcher(&reg);
    GroupedConvProblem problem = phase0_problem();

    float time = dispatcher.run(problem, nullptr);
    assert(time >= 0.0f);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_dispatcher_select()
{
    std::cout << "  test_grouped_conv_dispatcher_select... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    auto key = phase0_key();
    reg.register_kernel(key, executable(key));

    GroupedConvDispatcher dispatcher(&reg);
    GroupedConvProblem problem = phase0_problem();

    const auto* selected = dispatcher.select(problem);
    assert(selected != nullptr);
    assert(selected->name().find("grouped_conv_") != std::string::npos);
    assert(selected->matches(problem));

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_dispatcher_heuristic_uses_executable_candidate_gate()
{
    std::cout << "  test_grouped_conv_dispatcher_heuristic_uses_executable_candidate_gate... ";
    GroupedConvRegistry reg;
    auto key = phase0_key();

    GroupedConvKernelSet metadata;
    metadata.add("fp16", "nhwgc_gkyxc_nhwgk", "forward", key.tile_m, key.tile_n);
    assert(reg.register_metadata_set(metadata));

    auto unsupported_key = key;
    unsupported_key.tile_m++;
    assert(reg.register_kernel(unsupported_key,
                               executable(unsupported_key, false)));

    auto supported_key = key;
    supported_key.tile_m += 2;
    assert(reg.register_kernel(supported_key, executable(supported_key)));

    GroupedConvDispatcher dispatcher(&reg);
    dispatcher.set_strategy(GroupedConvDispatcher::SelectionStrategy::Heuristic);
    dispatcher.set_heuristic([](const auto&) {
        return std::vector<std::string>{"grouped_conv_fwd", "unsupported-id", "supported-id"};
    });

    const auto* selected = dispatcher.select(phase0_problem());
    assert(selected != nullptr);
    assert(selected->key().tile_m == supported_key.tile_m);
    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "\n=== Test Grouped Conv Registry ===\n\n";
    test_grouped_conv_registry_basic();
    test_grouped_conv_key_and_id();
    test_grouped_conv_executable_registration_validation();
    test_grouped_conv_supported_lookup();
    test_grouped_conv_registry_register_metadata_set();
    test_grouped_conv_registry_all_kernels();
    test_grouped_conv_registry_clear();
    test_grouped_conv_registry_thread_safe();
    test_grouped_conv_registry_export_json();
    test_grouped_conv_registry_filter();
    test_grouped_conv_dispatcher_basic();
    test_grouped_conv_dispatcher_select();
    test_grouped_conv_dispatcher_heuristic_uses_executable_candidate_gate();
    std::cout << "\n=== All Tests Passed! ===\n\n";
    return 0;
}
