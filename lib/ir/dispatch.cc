#include "triton/ir/dispatch.h"

namespace triton {
namespace ir {


[[ noreturn ]] void throw_unreachable(std::string key) {
  throw std::runtime_error("Encountered unimplemented code path in `" + key + "`. "
                           "This is likely a bug on our side.");
}

//===----------------------------------------------------------------------===//
//                              Programming Model
//===----------------------------------------------------------------------===//

ast::value *dispatch::program_id(int axis, ast::context *ctx, ir::builder *builder) {
  ir::value *ret = builder->create_get_program_id(axis);
  return ctx->create_value(ret);
}

ast::value *dispatch::num_programs(int axis, ast::context *ctx, ir::builder *builder) {
  ir::value *ret = builder->create_get_num_programs(axis);
  return ctx->create_value(ret);
}

//===----------------------------------------------------------------------===//
//                               Implicit Casting Utilities
//===----------------------------------------------------------------------===//

static ast::type *integer_promote(ast::type* a_ty, ast::type* b_ty){
  int a_rank = a_ty->get_integer_bitwidth();
  int b_rank = b_ty->get_integer_bitwidth();
  auto a_sn = a_ty->get_integer_signedness();
  auto b_sn = b_ty->get_integer_signedness();
  // Rules for signedness taken from "Usual arithmetic conversions" on
  // https://en.cppreference.com/w/c/language/conversion.
  if (a_sn == b_sn) {
    return a_rank > b_rank ? a_ty : b_ty;
  } else if (a_sn == ast::type::signedness::UNSIGNED) {
    return a_rank >= b_rank ? a_ty : b_ty;
  } else if (b_sn == ast::type::signedness::UNSIGNED) {
    return b_rank >= a_rank ? b_ty : a_ty;
  } else {
    throw_unreachable("integer_promote");
  }
}

enum class DivOrMod { NO, YES };

static ast::type *computation_type(ast::type* a_ty, ast::type* b_ty, DivOrMod div_or_mod) {
  ast::context &ctx = a_ty->get_context();
  // 1) if one operand is double, the other is implicitly
  //    converted to double
  if (a_ty->is_fp64_ty() || b_ty->is_fp64_ty())
    return ast::type::get_fp64_ty(ctx);
  // 2) if one operand is float, the other is implicitly
  //    converted to float
  if (a_ty->is_fp32_ty() || b_ty->is_fp32_ty())
    return ast::type::get_fp32_ty(ctx);
  // 3 ) if one operand is half, the other is implicitly converted to half
  //     unless we're doing / or %, which do not exist natively in PTX for fp16.
  if (a_ty->is_fp16_ty() || b_ty->is_fp16_ty()) {
    if (div_or_mod == DivOrMod::YES) {
      return ast::type::get_fp32_ty(ctx);
    } else {
      return ast::type::get_fp16_ty(ctx);
    }
  }
  if (!a_ty->is_integer_ty() || !b_ty->is_integer_ty())
    throw_unreachable("computation_type");
  // 4 ) both operands are integer and undergo
  //    integer promotion
  if (div_or_mod == DivOrMod::YES && a_ty->get_integer_signedness() != b_ty->get_integer_signedness()) {
    throw semantic_error("Cannot use /, //, or % with " + a_ty->repr() + " and " + b_ty->repr() + " because they have different signedness; this is unlikely to result in a useful answer. Cast them to the same signedness.");
  }
  return integer_promote(a_ty, b_ty);
}

//===----------------------------------------------------------------------===//
//                               Binary Operators
//===----------------------------------------------------------------------===//

static void throw_incompatible_types(ast::type* type_a, ast::type* type_b) {
  throw semantic_error("invalid operands of type " + type_a->repr() + " and " + type_b->repr());
}

static void check_ptr_type(ast::type* type_a, ast::type* type_b, bool allow_ptr_a){
  if(type_a->is_pointer_ty()){
    if(!allow_ptr_a)
      throw_incompatible_types(type_a, type_b);
    // T* + U* with T != U
    if(type_b->is_pointer_ty() && (type_a != type_b))
      throw_incompatible_types(type_a, type_b);
    // T* + float
    if(type_b->is_floating_point_ty())
      throw_incompatible_types(type_a, type_b);
  }
}

static void binary_op_type_checking(ast::value*& lhs, ast::value*& rhs, 
                             ast::context *ctx, ir::builder* builder,
                             bool allow_lhs_ptr = false, bool allow_rhs_ptr = false,
                             bool arithmetic_check = true, DivOrMod div_or_mod = DivOrMod::NO) {
  // implicit broadcasting
  std::tie(lhs, rhs) = dispatch::broadcast(lhs, rhs, ctx, builder);
  // implicit typecasting
  ast::type *lhs_sca_ty = lhs->get_type()->get_scalar_ty();
  ast::type *rhs_sca_ty = rhs->get_type()->get_scalar_ty();
  check_ptr_type(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr);
  check_ptr_type(rhs_sca_ty, lhs_sca_ty, allow_rhs_ptr);
  if (arithmetic_check && !lhs_sca_ty->is_pointer_ty() && !rhs_sca_ty->is_pointer_ty()) {
    ast::type *ret_sca_ty = computation_type(lhs_sca_ty, rhs_sca_ty, div_or_mod);
    lhs = dispatch::cast(lhs, ret_sca_ty, ctx, builder);
    rhs = dispatch::cast(rhs, ret_sca_ty, ctx, builder);
  }
}

ast::value *dispatch::add(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, ctx, builder, true, true);
  ast::type *ret_ty = input->get_type();
  ast::type *input_scalar_ty = input->get_type()->get_scalar_ty();
  ast::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  // offset + ptr
  // ptr + offset
  if(other_scalar_ty->is_pointer_ty() && !input_scalar_ty->is_pointer_ty())
    std::swap(input, other);
  if (input_scalar_ty->is_pointer_ty()) {
    ir::value *ret = builder->create_gep(input->get_ir_value(), {other->get_ir_value()});
    return ctx->create_value(ret, ret_ty);
  }
  // float + float
  else if (input_scalar_ty->is_floating_point_ty()) {
    ir::value *ret = builder->create_fadd(input->get_ir_value(), other->get_ir_value());
    return ctx->create_value(ret, ret_ty);
  }
  // int + int
  else if (input_scalar_ty->is_integer_ty()) {
    ir::value *ret = builder->create_add(input->get_ir_value(), other->get_ir_value());
    return ctx->create_value(ret, ret_ty);
  }
  throw_unreachable("add");
}

ast::value *dispatch::sub(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, ctx, builder, true, false);
  ast::type *ret_ty = input->get_type();
  ast::type *input_scalar_ty = input->get_type()->get_scalar_ty();
  ast::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  // ptr - offset
  if (input_scalar_ty->is_pointer_ty()) {
    ir::value *ret = builder->create_gep(input->get_ir_value(), {minus(other, ctx, builder)->get_ir_value()});
    return ctx->create_value(ret, ret_ty);
  }
  // float + float
  if (input_scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fsub(input->get_ir_value(), other->get_ir_value()), ret_ty);
  // int + int
  else if (input_scalar_ty->is_integer_ty())
    return ctx->create_value(builder->create_sub(input->get_ir_value(), other->get_ir_value()), ret_ty);
  throw_unreachable("sub");
}

ast::value *dispatch::mul(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, ctx, builder);
  ast::type *ret_ty = input->get_type();
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float * float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fmul(input->get_ir_value(), other->get_ir_value()), ret_ty);
  // int * int
  else if (scalar_ty->is_integer_ty())
    return ctx->create_value(builder->create_mul(input->get_ir_value(), other->get_ir_value()), ret_ty);
  throw_unreachable("mul");
}

ast::value *dispatch::truediv(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, ctx, builder, false, false, true, DivOrMod::YES);
  ast::type *input_scalar_ty = input->get_type()->get_scalar_ty();
  ast::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  // float / int
  if(input_scalar_ty->is_floating_point_ty() && other_scalar_ty->is_integer_ty())
    other = cast(other, input_scalar_ty, ctx, builder);
  // int / float
  else if(input_scalar_ty->is_integer_ty() && other_scalar_ty->is_floating_point_ty())
    input = cast(input, other_scalar_ty, ctx, builder);
  // int / int (cast to float32)
  else if(input_scalar_ty->is_integer_ty() && other_scalar_ty->is_integer_ty()){
    input = cast(input, ast::type::get_fp32_ty(*ctx), ctx, builder);
    other = cast(other, ast::type::get_fp32_ty(*ctx), ctx, builder);
  }
  // float / float (cast to highest exponent type)
  else if(input_scalar_ty->is_floating_point_ty() && other_scalar_ty->is_floating_point_ty()){
    if(input_scalar_ty->get_fp_mantissa_width() > other_scalar_ty->get_fp_mantissa_width())
      other = cast(other, input_scalar_ty, ctx, builder);
    else
      input = cast(input, other_scalar_ty, ctx, builder);
  }
  // unreachable
  else
    throw_unreachable("div");
  return ctx->create_value(builder->create_fdiv(input->get_ir_value(), other->get_ir_value()), input->get_type());
}

ast::value *dispatch::floordiv(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder){
  binary_op_type_checking(input, other, ctx, builder, false, false, true, DivOrMod::YES);
  ast::type *input_scalar_ty = input->get_type()->get_scalar_ty();
  ast::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  if(input_scalar_ty->is_integer_ty() && other_scalar_ty->is_integer_ty()){
    ast::type *ret_ty = integer_promote(input_scalar_ty, other_scalar_ty);
    input = cast(input, ret_ty, ctx, builder);
    other = cast(other, ret_ty, ctx, builder);
    if (ret_ty->is_integer_signed()) {
      return ctx->create_value(builder->create_sdiv(input->get_ir_value(), other->get_ir_value()), ret_ty);
    } else {
      return ctx->create_value(builder->create_udiv(input->get_ir_value(), other->get_ir_value()), ret_ty);
    }
  }
  throw_unreachable("floordiv");
}

ast::value *dispatch::fdiv(ast::value *input, ast::value *other, ir::constant_int *ieee_rounding, ast::context *ctx, ir::builder *builder){
  ast::type *input_scalar_ty = input->get_type()->get_scalar_ty();
  ast::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  if(!input_scalar_ty->is_floating_point_ty() || !other_scalar_ty->is_floating_point_ty())
    throw semantic_error("both operands of fdiv must have floating point scalar type");
  binary_op_type_checking(input, other, ctx, builder, false, false, false, DivOrMod::YES);
  ast::value* ret = ctx->create_value(builder->create_fdiv(input->get_ir_value(), other->get_ir_value()), input->get_type());
  if(ir::binary_operator* binop = dynamic_cast<ir::binary_operator*>(ret->get_ir_value()))
    binop->set_fdiv_ieee_rounding(ieee_rounding->get_value());
  return ret;
}

ast::value *dispatch::mod(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, ctx, builder, false, false, true, DivOrMod::YES);
  ast::type *ret_ty = input->get_type();
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  ast::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  // float % float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_frem(input->get_ir_value(), other->get_ir_value()), ret_ty);
  // int % int
  else if (scalar_ty->is_integer_ty()) {
    if (scalar_ty->get_integer_signedness() != other_scalar_ty->get_integer_signedness()) {
      throw semantic_error("Cannot mod " + scalar_ty->repr() + " by " + other_scalar_ty->repr() + " because they have different signedness; this is unlikely to result in a useful answer. Cast them to the same signedness.");
    }
    if (scalar_ty->is_integer_signed()) {
      return ctx->create_value(builder->create_srem(input->get_ir_value(), other->get_ir_value()), ret_ty);
    } else {
      return ctx->create_value(builder->create_urem(input->get_ir_value(), other->get_ir_value()), ret_ty);
    }
  }
  throw_unreachable("mod");
}


static void bitwise_op_type_checking(ast::value *&input, ast::value *&other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, ctx, builder, false, false, false);
  ast::type *input_sca_ty = input->get_type()->get_scalar_ty();
  ast::type *other_sca_ty = other->get_type()->get_scalar_ty();
  if(!input_sca_ty->is_integer_ty() || !other_sca_ty->is_integer_ty())
    throw_incompatible_types(input_sca_ty, other_sca_ty);
  ast::type *ret_sca_ty = integer_promote(input_sca_ty, other_sca_ty);
  if (ret_sca_ty != input_sca_ty)
    input = dispatch::cast(input, ret_sca_ty, ctx, builder);
  if (ret_sca_ty != other_sca_ty)
    other = dispatch::cast(other, ret_sca_ty, ctx, builder);
}

ast::value *dispatch::and_(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  bitwise_op_type_checking(input, other, ctx, builder);
  ast::type *ret_ty = input->get_type();
  return ctx->create_value(builder->create_and(input->get_ir_value(), other->get_ir_value()), ret_ty);
}

ast::value *dispatch::or_(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  bitwise_op_type_checking(input, other, ctx, builder);
  ast::type *ret_ty = input->get_type();
  return ctx->create_value(builder->create_or(input->get_ir_value(), other->get_ir_value()), ret_ty);
}


ast::value *dispatch::xor_(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  bitwise_op_type_checking(input, other, ctx, builder);
  ast::type *ret_ty = input->get_type();
  return ctx->create_value(builder->create_xor(input->get_ir_value(), other->get_ir_value()), ret_ty);
}


ast::value *dispatch::lshr(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  bitwise_op_type_checking(input, other, ctx, builder);
  ast::type *ret_ty = input->get_type();
  return ctx->create_value(builder->create_lshr(input->get_ir_value(), other->get_ir_value()), ret_ty);
}


ast::value *dispatch::shl(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  bitwise_op_type_checking(input, other, ctx, builder);
  ast::type *ret_ty = input->get_type();
  return ctx->create_value(builder->create_shl(input->get_ir_value(), other->get_ir_value()), ret_ty);
}

//===----------------------------------------------------------------------===//
//                               Unary Operators
//===----------------------------------------------------------------------===//

ast::value *dispatch::plus(ast::value *input, ast::context *ctx, ir::builder *) {
  return input;
}

ast::value *dispatch::minus(ast::value *input, ast::context *ctx, ir::builder *builder) {
  ast::type* input_sca_ty = input->get_type()->get_scalar_ty();
  if(input_sca_ty->is_pointer_ty())
    throw semantic_error("wrong type argument to unary minus (" + input_sca_ty->repr() + ")");
  ast::value *_0 = ctx->create_value(ir::constant::get_null_value(input_sca_ty->get_ir_type()), input_sca_ty);
  return dispatch::sub(_0, input, ctx, builder);
}

ast::value *dispatch::invert(ast::value *input, ast::context *ctx, ir::builder *builder) {
  ast::type* input_sca_ty = input->get_type()->get_scalar_ty();
  if(input_sca_ty->is_pointer_ty() || input_sca_ty->is_floating_point_ty())
    throw semantic_error("wrong type argument to unary invert (" + input_sca_ty->repr() + ")");
  ast::value *_1 = ctx->create_value(ir::constant::get_all_ones_value(input_sca_ty->get_ir_type()),
                                     input_sca_ty);
  return dispatch::xor_(input, _1, ctx, builder);
}


//===----------------------------------------------------------------------===//
//                               Comparison Operators
//===----------------------------------------------------------------------===//

ast::value *dispatch::greater_than(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, ctx, builder);
  ast::type *ret_ty = input->get_type();
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float > float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fcmpOGT(input->get_ir_value(), other->get_ir_value()), ret_ty);
  // int > int
  else if (scalar_ty->is_integer_ty()) {
    if (scalar_ty->is_integer_signed()) {
      return ctx->create_value(builder->create_icmpSGT(input->get_ir_value(), other->get_ir_value()), ret_ty);
    } else {
      return ctx->create_value(builder->create_icmpUGT(input->get_ir_value(), other->get_ir_value()), ret_ty);
    }
  }
  throw_unreachable("greater_than");
}

ast::value *dispatch::greater_equal(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, ctx, builder);
  ast::type *ret_ty = input->get_type();
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float >= float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fcmpOGE(input->get_ir_value(), other->get_ir_value()), ret_ty);
  // int >= int
  else if (scalar_ty->is_integer_ty()) {
    if (scalar_ty->is_integer_signed()) {
      return ctx->create_value(builder->create_icmpSGE(input->get_ir_value(), other->get_ir_value()), ret_ty);
    } else {
      return ctx->create_value(builder->create_icmpUGE(input->get_ir_value(), other->get_ir_value()), ret_ty);
    }
  }
  throw_unreachable("greater_equal");
}

ast::value *dispatch::less_than(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, ctx, builder);
  ast::type *ret_ty = input->get_type();
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float < float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fcmpOLT(input->get_ir_value(), other->get_ir_value()), ret_ty);
  // int < int
  else if (scalar_ty->is_integer_ty()) {
    if (scalar_ty->is_integer_signed())
      return ctx->create_value(builder->create_icmpSLT(input->get_ir_value(), other->get_ir_value()), ret_ty);
    else
      return ctx->create_value(builder->create_icmpULT(input->get_ir_value(), other->get_ir_value()), ret_ty);
  }
  throw_unreachable("less_than");
}

ast::value *dispatch::less_equal(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, ctx, builder);
  ast::type *ret_ty = input->get_type();
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float < float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fcmpOLE(input->get_ir_value(), other->get_ir_value()), ret_ty);
  // int < int
  else if (scalar_ty->is_integer_ty()) {
    if (scalar_ty->is_integer_signed())
      return ctx->create_value(builder->create_icmpSLE(input->get_ir_value(), other->get_ir_value()), ret_ty);
    else
      return ctx->create_value(builder->create_icmpULE(input->get_ir_value(), other->get_ir_value()), ret_ty);
  }
  throw_unreachable("less_equal");
}

ast::value *dispatch::equal(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, ctx, builder);
  ast::type *ret_ty = input->get_type();
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float == float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fcmpOEQ(input->get_ir_value(), other->get_ir_value()), ret_ty);
  // int == int
  else if (scalar_ty->is_integer_ty())
    return ctx->create_value(builder->create_icmpEQ(input->get_ir_value(), other->get_ir_value()), ret_ty);
  throw_unreachable("equal");
}

ast::value *dispatch::not_equal(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, ctx, builder);
  ast::type *ret_ty = input->get_type();
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float == float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fcmpUNE(input->get_ir_value(), other->get_ir_value()), ret_ty);
  // int == int
  else if (scalar_ty->is_integer_ty())
    return ctx->create_value(builder->create_icmpNE(input->get_ir_value(), other->get_ir_value()), ret_ty);
  throw_unreachable("equal");
}

//===----------------------------------------------------------------------===//
//                               Block Creation
//===----------------------------------------------------------------------===//

ast::value* dispatch::arange(int start, int end, ast::context *ctx, ir::builder *builder) {
  return ctx->create_value(builder->get_range(start, end));
}

ast::value* dispatch::zeros(shape_t shape, ast::type *dtype, ast::context *ctx, ir::builder *builder) {
  ir::value *_0 = ir::constant::get_null_value(dtype->get_ir_type());
  ir::value *ret = builder->create_splat(_0, shape);
  ast::type *ret_ty = ctx->get_type_from_ir(ret, dtype->get_integer_signedness());
  return ctx->create_value(ret, ret_ty);
}

//===----------------------------------------------------------------------===//
//                               Shape Manipulation
//===----------------------------------------------------------------------===//


ast::value *dispatch::reshape(ast::value *input, shape_t dst_shape, ast::context *ctx, ir::builder *builder) {
  unsigned numel = 1;
  for(unsigned s: dst_shape) numel *= s;
  if(input->get_type()->get_tile_num_elements() != numel)
    throw semantic_error("cannot reshape block of different shape");
  ir::value *ret = builder->create_reshape(input->get_ir_value(), dst_shape);
  ast::type *ret_ty = ctx->get_type_from_ir(ret, input->get_type()->get_integer_signedness());
  return ctx->create_value(ret, ret_ty);
}

ast::value *dispatch::cat(ast::value *lhs, ast::value *rhs, ast::context *ctx, ir::builder *builder) {
  // TODO: check semantic
  ir::value *ret = builder->create_cat(lhs->get_ir_value(), rhs->get_ir_value());
  ast::type *ret_ty = ctx->get_type_from_ir(ret, lhs->get_type()->get_integer_signedness());;
  return ctx->create_value(ret, ret_ty);
}

ast::value *dispatch::broadcast(ast::value *input, shape_t shape, ast::context *ctx, ir::builder *builder) {
  if (!input->get_type()->is_block_ty()) {
    ir::value *ret = builder->create_splat(input->get_ir_value(), shape);
    ast::type *ret_ty = ctx->get_type_from_ir(ret, input->get_type()->get_integer_signedness());
    return ctx->create_value(ret, ret_ty);
  }
  auto src_shape = input->get_type()->get_block_shapes();
  if (src_shape.size() != shape.size())
    throw std::runtime_error("Cannot broadcast");
  if(shape == src_shape)
    return input;
  ir::value *ret = builder->create_broadcast(input->get_ir_value(), shape);
  ast::type *ret_ty = ctx->get_type_from_ir(ret, input->get_type()->get_integer_signedness());
  return ctx->create_value(ret, ret_ty);
}

std::tuple<ast::value*, ast::value*> dispatch::broadcast(ast::value *lhs, ast::value* rhs, ast::context *ctx, ir::builder *builder) {
  ast::type *lhs_ty = lhs->get_type();
  ast::type *rhs_ty = rhs->get_type();

  // make_shape_compatible(block, scalar)
  if (lhs_ty->is_block_ty() && !rhs_ty->is_block_ty())
    rhs = ctx->create_value(builder->create_splat(rhs->get_ir_value(), lhs_ty->get_block_shapes()), rhs_ty);
  // make_shape_compatible(scalar, block)
  else if (!lhs_ty->is_block_ty() && rhs_ty->is_block_ty())
    lhs = ctx->create_value(builder->create_splat(lhs->get_ir_value(), rhs_ty->get_block_shapes()), lhs_ty);
  // make_shape_compatible(block, block)
  else if (lhs_ty->is_block_ty() && rhs_ty->is_block_ty()) {
    auto lhs_shape = lhs_ty->get_block_shapes();
    auto rhs_shape = rhs_ty->get_block_shapes();
    if (lhs_shape.size() != rhs_shape.size())
      throw std::runtime_error("Cannot make_shape_compatible: blocks must have the same rank");
    ir::type::block_shapes_t ret_shape;
    for (size_t i = 0; i < lhs_shape.size(); ++i) {
      unsigned left = lhs_shape[i];
      unsigned right = rhs_shape[i];
      if (left == 1)
        ret_shape.push_back(right);
      else if (right == 1)
        ret_shape.push_back(left);
      else if (left == right)
        ret_shape.push_back(left);
      else
        throw std::runtime_error("Cannot make_shape_compatible: incompatible dimensions at index " + std::to_string(i) +
                                 ": " + std::to_string(left) + " and " + std::to_string(right));
    }
    if (lhs_shape != ret_shape)
      lhs = ctx->create_value(builder->create_broadcast(lhs->get_ir_value(), ret_shape), lhs_ty);
    if (rhs_shape != ret_shape)
      rhs = ctx->create_value(builder->create_broadcast(rhs->get_ir_value(), ret_shape), rhs_ty);
  }
  return std::make_tuple(lhs, rhs);
}

ast::value *dispatch::bitcast(ast::value *input, ast::type *dst_ty, ast::context *ctx, ir::builder *builder){
  ast::type *src_ty = input->get_type();
  if (src_ty->is_block_ty())
    dst_ty = ctx->get_type_from_ir_type(ir::block_type::get(dst_ty->get_ir_type(), 
                                                       src_ty->get_block_shapes()),
                                        src_ty->get_integer_signedness());
  if(src_ty == dst_ty)
    return input;
  ast::type *src_sca_ty = src_ty->get_scalar_ty();
  ast::type *dst_sca_ty = dst_ty->get_scalar_ty();
  if(src_sca_ty->is_pointer_ty() || dst_sca_ty->is_pointer_ty())
    return cast(input, dst_ty, ctx, builder);
  // Bitcast
  int src_bits = src_sca_ty->get_primitive_size_in_bits();
  int dst_bits = dst_sca_ty->get_primitive_size_in_bits();
  if(src_bits != dst_bits)
    throw std::runtime_error("Cannot bitcast data-type of size " + std::to_string(src_bits) +
                             "to data-type of size " + std::to_string(dst_bits));
  return ctx->create_value(builder->create_cast(ir::BitCast, input->get_ir_value(), dst_ty->get_ir_type()), dst_ty);
}

ast::value *dispatch::cast(ast::value *input, ast::type *dst_ty, ast::context *ctx, ir::builder *builder) {
  ast::type *src_ty = input->get_type();
  if (src_ty->is_block_ty())
    dst_ty = ctx->get_type_from_ir_type(ir::block_type::get(dst_ty->get_ir_type(), 
                                                    src_ty->get_block_shapes()),
                                        src_ty->get_integer_signedness());
  if (src_ty == dst_ty)
    return input;
  ast::type *src_sca_ty = src_ty->get_scalar_ty();
  ast::type *dst_sca_ty = dst_ty->get_scalar_ty();
  // FP Truncation
  bool truncate_fp = src_sca_ty->is_floating_point_ty() &&
                     dst_sca_ty->is_floating_point_ty() &&
                     src_sca_ty->get_fp_mantissa_width() > dst_sca_ty->get_fp_mantissa_width();
  if (truncate_fp)
    return ctx->create_value(builder->create_fp_trunc(input->get_ir_value(), dst_ty->get_ir_type()), dst_ty);
  // FP Extension
  bool ext_fp = src_sca_ty->is_floating_point_ty() &&
                dst_sca_ty->is_floating_point_ty() &&
                src_sca_ty->get_fp_mantissa_width() < dst_sca_ty->get_fp_mantissa_width();
  if (ext_fp)
    return ctx->create_value(builder->create_fp_ext(input->get_ir_value(), dst_ty->get_ir_type()), dst_ty);
  // Int cast
  if (src_sca_ty->is_integer_ty() && dst_sca_ty->is_integer_ty() &&
      (src_sca_ty->get_integer_bitwidth() != dst_sca_ty->get_integer_bitwidth() ||
       src_sca_ty->get_integer_signedness() != dst_sca_ty->get_integer_signedness())) {
    bool sign_extend = src_sca_ty->is_integer_signed() && src_sca_ty->get_ir_type() != builder->get_int1_ty();
    return ctx->create_value(builder->create_int_cast(input->get_ir_value(), dst_ty->get_ir_type(), sign_extend), dst_ty);
  }
  // Float -> Int
  if (src_sca_ty->is_floating_point_ty() && dst_sca_ty->is_integer_ty()){
    if(dst_sca_ty->is_bool_ty())
      return ctx->create_value(builder->create_fp_to_ui(input->get_ir_value(), dst_ty->get_ir_type()), dst_ty);
    else
      return ctx->create_value(builder->create_fp_to_si(input->get_ir_value(), dst_ty->get_ir_type()), dst_ty);
  }
  // int -> Float
  if (src_sca_ty->is_integer_ty() && dst_sca_ty->is_floating_point_ty()){
    if (src_sca_ty->is_bool_ty() || !src_sca_ty->is_integer_signed())
      return ctx->create_value(builder->create_ui_to_fp(input->get_ir_value(), dst_ty->get_ir_type()), dst_ty);
    else
      return ctx->create_value(builder->create_si_to_fp(input->get_ir_value(), dst_ty->get_ir_type()), dst_ty);
  }
  // pointer -> int
  if (src_sca_ty->is_pointer_ty() && dst_sca_ty->is_integer_ty()){
    int bitwidth = dst_sca_ty->get_integer_bitwidth();
    if (bitwidth == 64)
      return ctx->create_value(builder->create_cast(ir::PtrToInt, input->get_ir_value(), dst_ty->get_ir_type()), dst_ty);
    if (bitwidth == 1)
      return dispatch::not_equal(dispatch::cast(input, 
                                                ctx->get_type_from_ir_type(builder->get_int64_ty()), 
                                                ctx, builder),
                                 ctx->create_value(builder->get_int64(0)),
                                 ctx,
                                 builder);
  }
  if (!src_sca_ty->is_pointer_ty() && dst_sca_ty->is_pointer_ty())
    return ctx->create_value(builder->create_cast(ir::IntToPtr, input->get_ir_value(), dst_ty->get_ir_type()), dst_ty);
  // Ptr -> Ptr
  if (src_sca_ty->is_pointer_ty() && dst_sca_ty->is_pointer_ty())
    return ctx->create_value(builder->create_cast(ir::BitCast, input->get_ir_value(), dst_ty->get_ir_type()), dst_ty);
  // * -> Bool
  if (dst_sca_ty->is_bool_ty()) {
    if (src_sca_ty->is_pointer_ty())
      input = cast(input, ctx->get_type_from_ir_type(builder->get_int64_ty()), ctx, builder);
    ast::value *other = ctx->create_value(builder->get_int64(0), ast::type::get_int64_ty(*ctx));
    if (src_ty->is_bool_ty())
      other = ctx->create_value(builder->create_splat(other->get_ir_value(), src_ty->get_block_shapes()), dst_ty);
    return ctx->create_value(builder->create_icmpNE(input->get_ir_value(), other->get_ir_value()), dst_ty);
  }
  throw_unreachable("casting from " + src_sca_ty->repr() + " to " + dst_sca_ty->repr());
}

//===----------------------------------------------------------------------===//
//                               Memory Operators
//===----------------------------------------------------------------------===//

ast::value *dispatch::load(ast::value* ptr, ast::value* mask, ast::value* other, 
                          const std::string &cache_modifier, int is_volatile, ast::context *ctx, ir::builder* builder) {
  if(!ptr->get_type()->get_scalar_ty()->is_pointer_ty())
    throw semantic_error("Pointer argument of load instruction is " + ptr->get_type()->repr());
  if(ptr->get_type()->is_block_ty()){
    if(mask){
      mask = dispatch::broadcast(mask, ptr->get_type()->get_block_shapes(), ctx, builder);
    }
    if(other){
      other = dispatch::broadcast(other, ptr->get_type()->get_block_shapes(), ctx, builder);
      other = dispatch::cast(other, ptr->get_type()->get_scalar_ty()->get_pointer_element_ty(), ctx, builder);
    }
  }
  ast::type *ptr_ty = ptr->get_type()->get_scalar_ty();
  ast::type *elt_ty = ptr_ty->get_pointer_element_ty();
  // treat bool* as int8*
  if (elt_ty == ast::type::get_int1_ty(*ctx)) {
    elt_ty = ast::type::get_int8_ty(*ctx);
    ptr_ty = ctx->get_type_from_ir_type(pointer_type::get(elt_ty->get_ir_type(), ptr_ty->get_pointer_address_space()));
    ptr = dispatch::cast(ptr, ptr_ty, ctx, builder);
  }
  // cache modifier
  load_inst::CACHE_MODIFIER cache = load_inst::NONE; // default
  if (!cache_modifier.empty()) {
    if (cache_modifier == ".ca")
      cache = load_inst::CA;
    else if (cache_modifier == ".cg")
      cache = load_inst::CG;
    else
      throw std::runtime_error(std::string("Cache modifier ") + cache_modifier + " not supported");
  }
  if (!mask && !other)
    return ctx->create_value(builder->create_load(ptr->get_ir_value(), cache, is_volatile), elt_ty);
  if (!mask)
    throw std::runtime_error("`other` cannot be provided without `mask`");
  auto shape = ptr->get_type()->get_block_shapes();
  if(!other){
    // FIXME: signedness info
    other = ctx->create_value(ir::undef_value::get(elt_ty->get_ir_type()));
    if(ptr->get_type()->is_block_ty())
      other = ctx->create_value(builder->create_splat(other->get_ir_value(), ptr->get_type()->get_block_shapes()));
  }
  return ctx->create_value(builder->create_masked_load(ptr->get_ir_value(), mask->get_ir_value(), other->get_ir_value(), cache, is_volatile),
                           elt_ty);
}

ast::value *dispatch::store(ast::value* ptr, ast::value *val, ast::value* mask, ast::context *ctx, ir::builder *builder) {
  if(!ptr->get_type()->get_scalar_ty()->is_pointer_ty())
    throw semantic_error("Pointer argument of store instruction is " + ptr->get_type()->repr());
  if(ptr->get_type()->is_block_ty())
    val = dispatch::broadcast(val, ptr->get_type()->get_block_shapes(), ctx, builder);
  if(mask)
    mask = dispatch::broadcast(mask, ptr->get_type()->get_block_shapes(), ctx, builder);
  ast::type *ptr_ty = ptr->get_type()->get_scalar_ty();
  ast::type *elt_ty = ptr_ty->get_pointer_element_ty();
  // treat bool* as int8*
  if (elt_ty == ast::type::get_int1_ty(*ctx)) {
    elt_ty = ast::type::get_int8_ty(*ctx);
    ptr_ty = ctx->get_type_from_ir_type(pointer_type::get(elt_ty->get_ir_type(), ptr_ty->get_pointer_address_space()));
    ptr = dispatch::cast(ptr, ptr_ty, ctx, builder);
  }
  // cast to target data-type
  val = dispatch::cast(val, elt_ty, ctx, builder);
  if (!mask)
    return ctx->create_value(builder->create_store(ptr->get_ir_value(), val->get_ir_value()));
  if(!mask->get_type()->get_scalar_ty()->is_bool_ty())
    throw semantic_error("Mask must have boolean scalar type");
  return ctx->create_value(builder->create_masked_store(ptr->get_ir_value(), val->get_ir_value(), mask->get_ir_value()));
}

ast::value *dispatch::atomic_cas(ast::value* ptr, ast::value *cmp, ast::value *val, ast::context *ctx, ir::builder *builder){
  return ctx->create_value(
    builder->create_atomic_cas(ptr->get_ir_value(), cmp->get_ir_value(), val->get_ir_value()),
    val->get_type()
  );
}

static void atom_red_typechecking(ast::value*& ptr, ast::value *&val, ast::value *&mask, ast::context *ctx, ir::builder *builder){
  if(!ptr->get_type()->get_scalar_ty()->is_pointer_ty())
    throw semantic_error("Pointer argument of store instruction is " + ptr->get_type()->repr());
  if(ptr->get_type()->is_block_ty()){
    if(mask){
      mask = dispatch::broadcast(mask, ptr->get_type()->get_block_shapes(), ctx, builder);
    }
    if(val){
      val = dispatch::broadcast(val, ptr->get_type()->get_block_shapes(), ctx, builder);
    }
  }
  val = dispatch::cast(val, ptr->get_type()->get_scalar_ty()->get_pointer_element_ty(), ctx, builder);
  if(!mask){
    mask = ctx->create_value(builder->get_int1(true));
    if(ptr->get_type()->is_block_ty())
      mask = ctx->create_value(builder->create_splat(mask->get_ir_value(), ptr->get_type()->get_block_shapes()),
                               val->get_type());
  }
}

ast::value *dispatch::atomic_max(ast::value* ptr, ast::value *val, ast::value *mask, ast::context *ctx, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, ctx, builder);
  ir::type* sca_ty = val->get_type()->get_scalar_ty();
  // direct call to atomic_max for integers
  if(sca_ty->is_integer_ty()) {
    if (sca_ty->is_integer_signed()) {
      return builder->create_atomic_rmw(ir::atomic_rmw_op_t::Max, ptr, val, mask);
    } else {
      return builder->create_atomic_rmw(ir::atomic_rmw_op_t::UMax, ptr, val, mask);
    }
  }
  // for float
  // return atomic_smax(i_ptr, i_val) if val >= 0
  // return atomic_umin(i_ptr, i_val) if val < 0
  ir::value* i_val = bitcast(val, builder->get_int32_ty(), builder);
  ir::value* i_ptr = bitcast(ptr, pointer_type::get(builder->get_int32_ty(), 1), builder);
  ir::value* pos = greater_equal(val, constant_fp::get(sca_ty, 0), builder);
  ir::value* neg = less_than(val, constant_fp::get(sca_ty, 0), builder);
  ir::value* pos_ret = builder->create_atomic_rmw(ir::atomic_rmw_op_t::Max, i_ptr, i_val, and_(mask, pos, builder));
  ir::value* neg_ret = builder->create_atomic_rmw(ir::atomic_rmw_op_t::UMin, i_ptr, i_val, and_(mask, neg, builder));
  return where(pos, pos_ret, neg_ret, ctx, builder);
}

ast::value *dispatch::atomic_min(ast::value* ptr, ast::value *val, ast::value *mask, ast::context *ctx, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, ctx, builder);
  ir::type* sca_ty = val->get_type()->get_scalar_ty();
  // direct call to atomic_min for integers
  if(sca_ty->is_integer_ty()) {
    if (sca_ty->is_integer_signed()) {
      return builder->create_atomic_rmw(ir::atomic_rmw_op_t::Min, ptr, val, mask);
    } else {
      return builder->create_atomic_rmw(ir::atomic_rmw_op_t::UMin, ptr, val, mask);
    }
  }
  // for float
  // return atomic_smin(i_ptr, i_val) if val >= 0
  // return atomic_umax(i_ptr, i_val) if val < 0
  ir::value* i_val = bitcast(val, builder->get_int32_ty(), builder);
  ir::value* i_ptr = bitcast(ptr, pointer_type::get(builder->get_int32_ty(), 1), builder);
  ir::value* pos = greater_equal(val, constant_fp::get(sca_ty, 0), builder);
  ir::value* neg = less_than(val, constant_fp::get(sca_ty, 0), builder);
  ir::value* pos_ret = builder->create_atomic_rmw(ir::atomic_rmw_op_t::Min, i_ptr, i_val, and_(mask, pos, builder));
  ir::value* neg_ret = builder->create_atomic_rmw(ir::atomic_rmw_op_t::UMax, i_ptr, i_val, and_(mask, neg, builder));
  return where(pos, pos_ret, neg_ret, builder);
}

ast::value *dispatch::atomic_add(ast::value* ptr, ast::value *val, ast::value *mask, ast::context *ctx, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, ctx, builder);
  ast::type* sca_ty = val->get_type()->get_scalar_ty();
  auto op = sca_ty->is_floating_point_ty() ? ir::atomic_rmw_op_t::FAdd : ir::atomic_rmw_op_t::Add;
  return ctx->create_value(builder->create_atomic_rmw(op, 
          ptr->get_ir_value(), val->get_ir_value(), mask->get_ir_value()), val->get_type());
}

ast::value *dispatch::atomic_and(ast::value* ptr, ast::value *val, ast::value *mask, ast::context *ctx, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, ctx, builder);
  return ctx->create_value(builder->create_atomic_rmw(ir::atomic_rmw_op_t::And, 
          ptr->get_ir_value(), val->get_ir_value(), mask->get_ir_value()), val->get_type());
}

ast::value *dispatch::atomic_or(ast::value* ptr, ast::value *val, ast::value *mask, ast::context *ctx, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, ctx, builder);
  return ctx->create_value(builder->create_atomic_rmw(ir::atomic_rmw_op_t::Or, 
          ptr->get_ir_value(), val->get_ir_value(), mask->get_ir_value()), val->get_type());
}

ast::value *dispatch::atomic_xor(ast::value* ptr, ast::value *val, ast::value *mask, ast::context *ctx, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, ctx, builder);
  return ctx->create_value(builder->create_atomic_rmw(ir::atomic_rmw_op_t::Xor, 
          ptr->get_ir_value(), val->get_ir_value(), mask->get_ir_value()), val->get_type());
}

ast::value *dispatch::atomic_xchg(ast::value* ptr, ast::value *val, ast::value *mask, ast::context *ctx, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, ctx, builder);
  ast::type* sca_ty = val->get_type()->get_scalar_ty();
  return ctx->create_value(builder->create_atomic_rmw(ir::atomic_rmw_op_t::Xchg, 
          ptr->get_ir_value(), val->get_ir_value(), mask->get_ir_value()), val->get_type());
}

//===----------------------------------------------------------------------===//
//                               Linear Algebra
//===----------------------------------------------------------------------===//

ast::value *dispatch::dot(ast::value *lhs, ast::value *rhs, ir::constant_int *allow_tf32, ast::context *ctx, ir::builder *builder) {
  ir::value *_0 = nullptr;
  if (lhs->get_type()->is_int_or_tileint_ty())
    _0 = builder->get_int32(0);
  else
    _0 = builder->get_float32(0);
  unsigned M = lhs->get_type()->get_block_shapes()[0];
  unsigned N = rhs->get_type()->get_block_shapes()[1];
  _0 = builder->create_splat(_0, {M, N});
  bool _allow_tf32 = allow_tf32->get_value() != 0;
  ir::value *ret = builder->create_dot(lhs->get_ir_value(), rhs->get_ir_value(), _0, _allow_tf32);
  return ctx->create_value(ret);
}


//===----------------------------------------------------------------------===//
//                               Indexing
//===----------------------------------------------------------------------===//

ast::value *dispatch::where(ast::value* condition, ast::value *x, ast::value *y, ast::context *ctx, ir::builder *builder){
  condition = dispatch::cast(condition, builder->get_int1_ty(), ctx, builder);
  if(condition->get_type()->is_block_ty()){
    x = dispatch::broadcast(x, condition->get_type()->get_block_shapes(), ctx, builder);
    y = dispatch::broadcast(y, condition->get_type()->get_block_shapes(), ctx, builder);
  }
  ast::type* x_ty = x->get_type()->get_scalar_ty();
  ast::type* y_ty = y->get_type()->get_scalar_ty();
  ast::type* ty = computation_type(x_ty, y_ty, DivOrMod::NO);
  x = dispatch::cast(x, ty, ctx, builder);
  y = dispatch::cast(y, ty, ctx, builder);
  ir::value *ret = builder->create_select(condition->get_ir_value(), x->get_ir_value(), y->get_ir_value());
  ast::type *ret_ty = ctx->get_type_from_ir(ret, ty->get_integer_signedness());
  return ctx->create_value(ret, ret_ty);
}


//===----------------------------------------------------------------------===//
//                               Reductions
//===----------------------------------------------------------------------===//

static ast::value *reduce_impl(ast::value *input, unsigned int axis, ast::context *ctx, ir::builder *builder, const std::string &name,
                       ir::reduce_inst::op_t FLOAT_OP, ir::reduce_inst::op_t INT_OP) {
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // input is extended to 32-bits if necessary
  // this increases numerical accuracy and can be done pretty much for free
  // on GPUs
  if(scalar_ty->is_integer_ty() && scalar_ty->get_integer_bitwidth() <= 32)
    input = dispatch::cast(input, ast::type::get_int32_ty(scalar_ty->get_context()), ctx, builder);
  if (scalar_ty->is_floating_point_ty()) {
    ir::value *ret = builder->create_reduce(input->get_ir_value(), FLOAT_OP, axis);
    ast::type *ret_ty = ctx->get_type_from_ir(ret, input->get_type()->get_integer_signedness());
    return ctx->create_value(ret, ret_ty);
  } else if (scalar_ty->is_integer_ty()) {
    ir::value *ret = builder->create_reduce(input->get_ir_value(), INT_OP, axis);
    ast::type *ret_ty = ctx->get_type_from_ir(ret, input->get_type()->get_integer_signedness());
    return ctx->create_value(ret, ret_ty);
  }
  throw_unreachable(name);
}

ast::value *dispatch::min(ast::value *input, unsigned int axis, ast::context *ctx, ir::builder *builder) {
  return reduce_impl(input, axis, ctx, builder, "min", ir::reduce_inst::FMIN, ir::reduce_inst::MIN);
}

ast::value *dispatch::max(ast::value *input, unsigned int axis, ast::context *ctx, ir::builder *builder) {
  return reduce_impl(input, axis, ctx, builder, "max", ir::reduce_inst::FMAX, ir::reduce_inst::MAX);
}

ast::value *dispatch::sum(ast::value *input, unsigned int axis, ast::context *ctx, ir::builder *builder) {
  return reduce_impl(input, axis, ctx, builder, "sum", ir::reduce_inst::FADD, ir::reduce_inst::ADD);
}

ast::value *dispatch::xor_sum(ast::value *input, unsigned int axis, ast::context *ctx, ir::builder *builder) {
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  if (!scalar_ty->is_integer_ty())
    throw semantic_error("xor_sum only supported for integers");
  return reduce_impl(input, axis, ctx, builder, "sum", ir::reduce_inst::XOR, ir::reduce_inst::XOR);
}


//===----------------------------------------------------------------------===//
//                               Math
//===----------------------------------------------------------------------===//

ast::value *dispatch::umulhi(ast::value *x, ast::value* y, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(x, y, ctx, builder);
  ir::value *ret = builder->insert(umulhi_inst::create(x->get_ir_value(), y->get_ir_value()));
  return ctx->create_value(ret, x->get_type());
}

ast::value *dispatch::exp(ast::value *x, ast::context *ctx, ir::builder *builder) {
  ir::value *ret = builder->create_exp(x->get_ir_value());
  ast::type *ret_ty = x->get_type();
  return ctx->create_value(ret, ret_ty);
}

ast::value *dispatch::log(ast::value *x, ast::context *ctx, ir::builder *builder) {
  ir::value *ret = builder->create_log(x->get_ir_value());
  ast::type *ret_ty = x->get_type();
  return ctx->create_value(ret, ret_ty);
}

ast::value *dispatch::cos(ast::value *x, ast::context *ctx, ir::builder *builder) {
  ir::value *ret = builder->create_cos(x->get_ir_value());
  ast::type *ret_ty = x->get_type();
  return ctx->create_value(ret, ret_ty);
}

ast::value *dispatch::sin(ast::value *x, ast::context *ctx, ir::builder *builder) {
  ir::value *ret = builder->create_sin(x->get_ir_value());
  ast::type *ret_ty = x->get_type();
  return ctx->create_value(ret, ret_ty);
}

ast::value *dispatch::sqrt(ast::value *x, ast::context *ctx, ir::builder *builder) {
  ir::value *ret = builder->create_sqrt(x->get_ir_value());
  ast::type *ret_ty = x->get_type();
  return ctx->create_value(ret, ret_ty);
}


//

ast::value *dispatch::multiple_of(ast::value *x, int value, ast::context *ctx, ir::builder *){
  ir::instruction* i = dynamic_cast<ir::instruction*>(x->get_ir_value());
  if(!i)
    throw_unreachable("multiple_of");
  i->set_metadata(ir::metadata::multiple_of, value);
  return x;
}

ast::value *dispatch::max_contiguous(ast::value *x, int value, ast::context *ctx, ir::builder *){
  ir::instruction* i = dynamic_cast<ir::instruction*>(x->get_ir_value());
  if(!i)
    throw_unreachable("max_contiguous");
  i->set_metadata(ir::metadata::max_contiguous, value);
  return x;
}

ast::value *dispatch::debug_barrier(ast::context *ctx, ir::builder *builder) {
  ir::value *ret = builder->create_barrier();
  ast::type *ret_ty = ctx->get_type_from_ir(ret);
  return ctx->create_value(ret, ret_ty);
}


}
}
