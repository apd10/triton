#ifndef _TRITON_AST_TYPE_H_
#define _TRITON_AST_TYPE_H_

//===----------------------------------------------------------------------===//
//
/// \file
/// Frontend types. Will be lowered to IR types by dispatch.cc
//
//===----------------------------------------------------------------------===//

#include <vector>
#include <string>

namespace triton {
namespace ast {

class context;

/// ast::type encapusulates ir::type and other type info (e.g., signedness)
class type {
public:
  enum class signedness { SIGNED, UNSIGNED };

private:
  ir::type *ir_ty_;
  signedness signedness_ = signedness::SIGNED;
  context &ctx; //< ast::context from where this type is generated

public:
  context &get_context() const { return ctx; }
  ir::type *get_ir_type() { return ir_ty_; }
  bool is_signed_integer() const;
  bool is_unsigned_integer() const;

public:
  // Factory methods
  static type *get_void_ty(context &ctx);
  static type *get_label_ty(context &ctx);
  static type *get_fp8_ty(context &ctx);
  static type *get_fp16_ty(context &ctx);
  static type *get_bf16_ty(context &ctx);
  static type *get_fp32_ty(context &ctx);
  static type *get_fp64_ty(context &ctx);
  // int
  static type *get_int1_ty(context &ctx);
  static type *get_int8_ty(context &ctx);
  static type *get_int16_ty(context &ctx);
  static type *get_int32_ty(context &ctx);
  static type *get_int64_ty(context &ctx);
  static type *get_int128_ty(context &ctx);
  // uint
  static type *get_uint8_ty(context &ctx);
  static type *get_uint16_ty(context &ctx);
  static type *get_uint32_ty(context &ctx);
  static type *get_uint64_ty(context &ctx);
  static type *get_uint128_ty(context &ctx);
  // derived type
  static type *get_pointer_ty(type *pointee_ty, unsigned address_space);
  static type *get_function_ty(type *ret_ty, const std::vector<type*>& param_tys);
  static type *get_block_ty(type *ty, const std::vector<unsigned> &shapes);

  // type attributes
  unsigned get_fp_mantissa_width() const { return ir_ty_->get_fp_mantissa_width(); }
  unsigned get_integer_bitwidth() const { return ir_ty_->get_integer_bitwidth(); }
  signedness get_integer_signedness() const { return signedness_; }
  bool is_integer_signed() const;
  unsigned get_tile_bitwidth() const { return ir_ty_->get_tile_bitwidth(); }
  unsigned get_primitive_size_in_bits() const { return ir_ty_->get_primitive_size_in_bits(); }
  type *get_scalar_ty() const;
  std::vector<unsigned> get_block_shapes() const { return ir_ty_->get_block_shapes(); }
  const size_t get_tile_rank() const { return ir_ty_->get_tile_rank(); }
  const size_t get_tile_ranks1() const { return ir_ty_->get_tile_ranks1(); }
  unsigned get_tile_num_elements() const { return ir_ty_->get_tile_num_elements(); }
  // TODO: fill this

  // primitive predicates
  bool is_void_ty() const { return ir_ty_->is_void_ty(); }
  bool is_fp8_ty() const  { return ir_ty_->is_fp8_ty(); }
  bool is_fp16_ty() const { return ir_ty_->is_fp16_ty(); }
  bool is_bf16_ty() const { return ir_ty_->is_bf16_ty(); }
  bool is_fp32_ty() const { return ir_ty_->is_fp32_ty(); }
  bool is_fp64_ty() const { return ir_ty_->is_fp64_ty(); }
  bool is_label_ty() const { return ir_ty_->is_label_ty(); }
  bool is_metadata_ty() const { return ir_ty_->is_metadata_ty(); }
  bool is_token_ty() const { return ir_ty_->is_token_ty(); }
  bool is_integer_ty() const { return ir_ty_->is_integer_ty(); }
  bool is_integer_ty(unsigned bitwidth, signedness sn) {
    return is_integer_ty() && get_integer_bitwidth() == bitwidth && signedness_ == sn;
  }
  bool is_bool_ty() const { return ir_ty_->is_bool_ty(); }
  bool is_pointer_ty() const { return ir_ty_->is_pointer_ty(); }
  bool is_block_ty() const { return ir_ty_->is_block_ty(); }

  // Composite predicates
  bool is_int_or_tileint_ty() { return ir_ty_->is_int_or_tileint_ty(); }
  bool is_integer_ty(unsigned width) { return ir_ty_->is_integer_ty(); }
  bool is_floating_point_ty() const { return ir_ty_->is_floating_point_ty(); }
  bool is_sized() const { return ir_ty_->is_sized(); }

  // repr
  std::string repr() const { return ir_ty_->repr(); }
};

} // namespace ast
} // namespace triton


#endif // _TRITON_AST_TYPE_H_
