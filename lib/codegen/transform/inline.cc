#include <iostream>
#include "triton/codegen/transform/inline.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"

namespace triton{
namespace codegen{
namespace transform{

void inliner::do_inline(ir::function* fn, ir::call_inst* callsite, ir::builder& builder,
                        std::map<ir::function*, std::vector<ir::call_inst*>>& callsites){
  ir::basic_block* parent_block = callsite->get_parent();
  ir::function* parent_fn = parent_block->get_parent();
   // the parent block is split into block A and block B:
  //   - block A (`new_blocks[0]`) is the entry block of the inlined function
  //   - block B (`exit`) resumes execution of the parent function
  ir::basic_block* entry = parent_block->split_before(callsite, fn->get_name());
  ir::basic_block* exit = entry->get_successors()[0];
  std::vector<ir::basic_block*> new_blocks = {entry};
  for(size_t i = 1; i < fn->blocks().size(); i++){
   ir::basic_block* block = fn->blocks()[i];
   ir::context& ctx = block->get_context();
   const std::string& name = block->get_parent()->get_name() + "_" + block->get_name();
   new_blocks.push_back(ir::basic_block::create(ctx, name, parent_fn));
  }
  // a phi node holds the return values of the inlined function
  builder.set_insert_point(exit->get_first_non_phi());
  ir::phi_node* exit_val = builder.create_phi(fn->get_fn_type()->get_return_ty(), 0);
  // get arguments `fn` is called with
  std::vector<ir::value*> tgt_args(callsite->op_begin(), callsite->op_end());
  std::vector<ir::argument*> src_args(fn->args().begin(), fn->args().end());
  // Actually generate the instructions:
  // - Remove the branch created by basic_block::split_before
  // - Clone all instructions
  // - Replace `ret` with incoming nodes to `exit_val` and branches to `exit`
  ir::instruction* terminator = new_blocks[0]->get_inst_list().back();
//  new_blocks[0]->get_inst_list().back()->erase_from_parent();
  terminator->erase_from_parent();
  for(size_t i = 0; i < new_blocks.size(); i++){
    ir::basic_block* old_block = fn->blocks()[i];
    ir::basic_block* new_block = new_blocks[i];
    builder.set_insert_point(new_block);
    for(ir::instruction* old_inst: old_block->get_inst_list()){
       // `ret` instruction is a special case:
      // instead of returning we need to branch to after the function call
      ir::instruction* new_inst = nullptr;
      if(ir::return_inst* ret = dynamic_cast<ir::return_inst*>(old_inst)){
        new_inst = ir::branch_inst::create(exit);
        if(ir::value* ret_val = ret->get_return_value())
          exit_val->add_incoming(ret_val, new_block);
      }
      else
        new_inst = old_inst->clone();
      if(ir::call_inst* call = dynamic_cast<ir::call_inst*>(new_inst)){
        callsites[call->get_fn()].push_back(call);
      }
      for(size_t k = 0; k < new_blocks.size(); k++)
        new_inst->replace_uses_of_with(fn->blocks()[k], new_blocks[k]);
      for(size_t k = 0; k < src_args.size(); k++)
        new_inst->replace_uses_of_with(src_args[k], tgt_args[k]);
      builder.insert(new_inst);
    }
  }
  // done -- make sure insert point is properly set to exit block
  builder.set_insert_point(exit);
}

void inliner::run(ir::module &mod) {

  // gather all call sites
  std::map<ir::function*, std::vector<ir::call_inst*>> callsites;
  for(ir::function* fn: mod.get_function_list())
  for(ir::basic_block* block: fn->blocks())
  for(ir::instruction* instr: block->get_inst_list())
  if(ir::call_inst* call = dynamic_cast<ir::call_inst*>(instr)){
    callsites[call->get_fn()].push_back(call);
  }

  // replace call sites with function bodies, one by one
  for(auto& x: callsites){
    ir::function* fn = x.first;
    for(ir::call_inst* callsite: x.second)
      do_inline(fn, callsite, mod.get_builder(), callsites);
    mod.remove_function(fn);
  }
}

}
}
}