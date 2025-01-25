/* Copyright 2024 The OpenXLA Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/transforms/collective_permute_valid_iteration_annotator.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class CollectivePermuteValidIterationAnnotatorTest
    : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<std::string> Transform(absl::string_view hlo,
                                        bool expect_change = true,
                                        FixedMapping params = {}) {
    // WhileLoopTripCountAnnotator should discover the while loop.
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        RunAndCheckHloRewrite(
                            hlo, WhileLoopTripCountAnnotator(), true, params));
    TF_ASSIGN_OR_RETURN(
        bool changed,
        CollectivePermuteValidIterationAnnotator().Run(module.get()));
    VLOG(0) << "HLO after CollectivePermuteValidIterationAnnotator: \n"
            << module->ToString(HloPrintOptions::ShortParsable()
                                    .set_print_control_dependencies(true));
    EXPECT_EQ(changed, expect_change);

    HloCollectivePermuteInstruction* cp =
        DynCastOrNull<HloCollectivePermuteInstruction>(
            FindInstruction(module.get(), HloOpcode::kCollectivePermute));
    if (cp == nullptr) {
      return absl::InternalError("No collective permute instruction found");
    }
    auto it = cp->frontend_attributes().map().find(kSendRecvValidationAttr);
    bool has_validation_attribute = it != cp->frontend_attributes().map().end();
    EXPECT_EQ(expect_change, has_validation_attribute);
    return has_validation_attribute ? it->second : "";
  }
};

const char* base_hlo = R"(
    HloModule test, entry_computation_layout={()->(s32[], s32[])}
    %Body (param: (s32[], s32[])) -> (s32[], s32[]) {
      %param = (s32[], s32[]) parameter(0)
      %i = s32[] get-tuple-element((s32[], s32[]) %param), index=1
      %one = s32[] constant(1)
      %i_plus_one = s32[] add(s32[] %i, s32[] %one)
      %permute = s32[] collective-permute(%i_plus_one), channel_id=1,
        source_target_pairs=$source_target_pairs
      ROOT %tuple = (s32[], s32[]) tuple(s32[] %permute, s32[] %i_plus_one)
    }
    %Cond (param.1: (s32[], s32[])) -> pred[] {
      %param.1 = (s32[], s32[]) parameter(0)
      %i.1 = s32[] get-tuple-element((s32[], s32[]) %param.1), index=1
      %trip_count = s32[] constant(10)
      ROOT %done = pred[] compare(s32[] %i.1, s32[] %trip_count), direction=LT
    }
    ENTRY %test () -> (s32[], s32[]) {
      %i_start = s32[] constant(0)
      %p_start = s32[] constant(0)
      %initial_tuple = (s32[], s32[]) tuple(s32[] %i_start, s32[] %p_start)
      ROOT %while = (s32[], s32[]) while((s32[], s32[]) %initial_tuple),
        condition=%Cond, body=%Body,
        frontend_attributes={$is_pipelined_while_loop}
    }
  )";

TEST_F(CollectivePermuteValidIterationAnnotatorTest, NoChange) {
  // We expect no changes here because the while loop is not labelled as
  // `is_pipelined_while_loop`.
  TF_ASSERT_OK(Transform(base_hlo, false,
                         {{"$source_target_pairs", "{{0,1},{1,2},{2,3},{3,0}}"},
                          {"$is_pipelined_while_loop", ""}}));
}

TEST_F(CollectivePermuteValidIterationAnnotatorTest, ForwardCycle) {
  FixedMapping params = {
      {"$source_target_pairs", "{{0,1},{1,2},{2,3},{3,0}}"},
      {"$is_pipelined_while_loop", "is_pipelined_while_loop=\"true\""}};
  TF_ASSERT_OK_AND_ASSIGN(std::string validation_attr,
                          Transform(base_hlo, true, params));
  EXPECT_EQ(validation_attr, "{{0,6},{1,7},{2,8},{3,9}}");
}

TEST_F(CollectivePermuteValidIterationAnnotatorTest, BackwardCycle) {
  FixedMapping params = {
      {"$source_target_pairs", "{{0,3},{1,0},{2,1},{3,2}}"},
      {"$is_pipelined_while_loop", "is_pipelined_while_loop=\"true\""}};
  TF_ASSERT_OK_AND_ASSIGN(std::string validation_attr,
                          Transform(base_hlo, true, params));
  EXPECT_EQ(validation_attr, "{{3,9},{2,8},{1,7},{0,6}}");
}
}  // namespace
}  // namespace xla
