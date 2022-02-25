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
  return ctx->create_value(builder->create_get_program_id(axis), ctx->get_int32_ty());
}

ast::value *dispatch::num_programs(int axis, ast::context *ctx, ir::builder *builder) {
  return ctx->create_value(builder->create_get_num_programs(axis), ctx->get_int32_ty());
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

static void binary_op_type_checking(ast::value*& lhs, ast::value*& rhs, ir::builder* builder,
                             bool allow_lhs_ptr = false, bool allow_rhs_ptr = false,
                             bool arithmetic_check = true, DivOrMod div_or_mod = DivOrMod::NO) {
  // implicit broadcasting
  std::tie(lhs, rhs) = dispatch::broadcast(lhs, rhs, builder);
  // implicit typecasting
  ast::type *lhs_sca_ty = lhs->get_type()->get_scalar_ty();
  ast::type *rhs_sca_ty = rhs->get_type()->get_scalar_ty();
  check_ptr_type(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr);
  check_ptr_type(rhs_sca_ty, lhs_sca_ty, allow_rhs_ptr);
  if (arithmetic_check && !lhs_sca_ty->is_pointer_ty() && !rhs_sca_ty->is_pointer_ty()) {
    ast::type *ret_sca_ty = computation_type(lhs_sca_ty, rhs_sca_ty, div_or_mod);
    lhs = dispatch::cast(lhs, ret_sca_ty, builder);
    rhs = dispatch::cast(rhs, ret_sca_ty, builder);
  }
}

ast::value *dispatch::add(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, builder, true, true);
  ast::type *input_scalar_ty = input->get_type()->get_scalar_ty();
  ast::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  // offset + ptr
  // ptr + offset
  if(other_scalar_ty->is_pointer_ty() && !input_scalar_ty->is_pointer_ty())
    std::swap(input, other);
  if (input_scalar_ty->is_pointer_ty())
    return ctx->create_value(builder->create_gep(input, {other}), input_scalar_ty);
  // float + float
  else if (input_scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fadd(input, other), input_scalar_ty);
  // int + int
  else if (input_scalar_ty->is_integer_ty())
    return ctx->create_value(builder->create_add(input, other), input_scalar_ty);
  throw_unreachable("add");
}

ast::value *dispatch::sub(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, builder, true, false);
  ast::type *input_scalar_ty = input->get_type()->get_scalar_ty();
  ast::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  // ptr - offset
  if (scalar_ty->is_pointer_ty())
    return ctx->create_value(builder->create_gep(input, {dispatch::minus(other, builder)}), input_scalar_ty);
  // float + float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fsub(input, other), input_scalar_ty);
  // int + int
  else if (scalar_ty->is_integer_ty())
    return ctx->create_value(builder->create_sub(input, other), input_scalar_ty);
  throw_unreachable("sub");
}

ast::value *dispatch::mul(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float * float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fmul(input, other), scalar_ty);
  // int * int
  else if (scalar_ty->is_integer_ty())
    return ctx->create_value(builder->create_mul(input, other), scalar_ty);
  throw_unreachable("mul");
}

ast::value *dispatch::truediv(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, builder, false, false, true, DivOrMod::YES);
  ast::type *input_scalar_ty = input->get_type()->get_scalar_ty();
  ast::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  // float / int
  if(input_scalar_ty->is_floating_point_ty() && other_scalar_ty->is_integer_ty())
    other = cast(other, input_scalar_ty, builder);
  // int / float
  else if(input_scalar_ty->is_integer_ty() && other_scalar_ty->is_floating_point_ty())
    input = cast(input, other_scalar_ty, builder);
  // int / int (cast to float32)
  else if(input_scalar_ty->is_integer_ty() && other_scalar_ty->is_integer_ty()){
    input = cast(input, builder->get_float_ty(), builder);
    other = cast(other, builder->get_float_ty(), builder);
  }
  // float / float (cast to highest exponent type)
  else if(input_scalar_ty->is_floating_point_ty() && other_scalar_ty->is_floating_point_ty()){
    if(input_scalar_ty->get_fp_mantissa_width() > other_scalar_ty->get_fp_mantissa_width())
      other = cast(other, input_scalar_ty, builder);
    else
      input = cast(input, other_scalar_ty, builder);
  }
  // unreachable
  else
    throw_unreachable("div");
  return ctx->create_value(builder->create_fdiv(input, other), input_scalar_ty);
}

ast::value *dispatch::floordiv(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder){
  binary_op_type_checking(input, other, builder, false, false, true, DivOrMod::YES);
  ast::type *input_scalar_ty = input->get_type()->get_scalar_ty();
  ast::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  if(input_scalar_ty->is_integer_ty() && other_scalar_ty->is_integer_ty()){
    ast::type *ret_ty = integer_promote(input_scalar_ty, other_scalar_ty);
    input = cast(input, ret_ty, builder);
    other = cast(other, ret_ty, builder);
    if (ret_ty->is_integer_signed()) {
      return ctx->create_value(builder->create_sdiv(input, other), input_scalar_ty);
    } else {
      return ctx->create_value(builder->create_udiv(input, other), input_scalar_ty);
    }
  }
  throw_unreachable("floordiv");
}

ast::value *dispatch::fdiv(ast::value *input, ast::value *other, constant_int *ieee_rounding, ast::context *ctx, ir::builder *builder){
  ast::type *input_scalar_ty = input->get_type()->get_scalar_ty();
  ast::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  if(!input_scalar_ty->is_floating_point_ty() || !other_scalar_ty->is_floating_point_ty())
    throw semantic_error("both operands of fdiv must have floating point scalar type");
  binary_op_type_checking(input, other, builder, false, false, false, DivOrMod::YES);
  ast::value* ret = ctx->create_value(builder->create_fdiv(input, other), input_scalar_ty);
  if(ir::binary_operator* binop = dynamic_cast<ir::binary_operator*>(ret->get_ir_value()))
    binop->set_fdiv_ieee_rounding(ieee_rounding->get_value());
  return ret;
}

ast::value *dispatch::mod(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, builder, false, false, true, DivOrMod::YES);
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  ast::type *other_scalar_ty = other->get_type()->get_scalar_ty();
  // float % int
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_frem(input, other), scalar_ty);
  // int % int
  else if (scalar_ty->is_integer_ty()) {
    if (scalar_ty->get_integer_signedness() != other_scalar_ty->get_integer_signedness()) {
      throw semantic_error("Cannot mod " + scalar_ty->repr() + " by " + other_scalar_ty->repr() + " because they have different signedness; this is unlikely to result in a useful answer. Cast them to the same signedness.");
    }
    if (scalar_ty->is_integer_signed()) {
      return ctx->(builder->create_srem(input, other), scalar_ty);
    } else {
      return ctx->(builder->create_urem(input, other), scalar_ty);
    }
  }
  throw_unreachable("mod");
}


static ast::type *bitwise_op_type_checking(ast::value *&input, ast::value *&other, ir::builder *builder) {
  binary_op_type_checking(input, other, builder, false, false, false);
  ast::type *input_sca_ty = input->get_type()->get_scalar_ty();
  ast::type *other_sca_ty = other->get_type()->get_scalar_ty();
  if(!input_sca_ty->is_integer_ty() || !other_sca_ty->is_integer_ty())
    throw_incompatible_types(input_sca_ty, other_sca_ty);
  ast::type *ret_sca_ty = integer_promote(input_sca_ty, other_sca_ty);
  if (ret_sca_ty != input_sca_ty)
    input = cast(input, ret_sca_ty, builder);
  if (ret_sca_ty != other_sca_ty)
    other = cast(other, ret_sca_ty, builder);
  return ret_sca_ty;
}

ast::value *dispatch::and_(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  ast::type *ret_ty = bitwise_op_type_checking(input, other, builder);
  return ctx->create_value(builder->create_and(input, other), ret_ty);
}

ast::value *dispatch::or_(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  ast::type *ret_ty = bitwise_op_type_checking(input, other, builder);
  return ctx->create_value(builder->create_or(input, other), ret_ty);
}


ast::value *dispatch::xor_(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  ast::type *ret_ty = bitwise_op_type_checking(input, other, builder);
  return ctx->create_value(builder->create_xor(input, other), ret_ty);
}


ast::value *dispatch::lshr(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  ast::type *ret_ty = bitwise_op_type_checking(input, other, builder);
  return ctx->create_value(builder->create_lshr(input, other), ret_ty);
}


ast::value *dispatch::shl(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  ast::type *ret_ty = bitwise_op_type_checking(input, other, builder);
  return ctx->create_value(builder->create_shl(input, other), ret_ty);
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
  ast::value *_0 = ctx->create_value(ir::constant::get_null_value(input_sca_ty), input_sca_ty);
  return ctx->create_value(dispatch::sub(_0, input, builder), input_sca_ty);
}

ast::value *dispatch::invert(ast::value *input, ast::context *ctx, ir::builder *builder) {
  ast::type* input_sca_ty = input->get_type()->get_scalar_ty();
  if(input_sca_ty->is_pointer_ty() || input_sca_ty->is_floating_point_ty())
    throw semantic_error("wrong type argument to unary invert (" + input_sca_ty->repr() + ")");
  ast::value *_1 = ctx->create_value(ir::constant::get_all_ones_value(input_sca_ty), input_sca_ty);
  return ctx->create_value(dispatch::xor_(input, _1, builder), input_sca_ty);
}


//===----------------------------------------------------------------------===//
//                               Comparison Operators
//===----------------------------------------------------------------------===//

ast::value *dispatch::greater_than(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float > float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fcmpOGT(input, other), scalar_ty);
  // int > int
  else if (scalar_ty->is_integer_ty()) {
    if (scalar_ty->is_integer_signed()) {
      return ctx->create_value(builder->create_icmpSGT(input, other), scalar_ty);
    } else {
      return ctx->create_value(builder->create_icmpUGT(input, other), scalar_ty);
    }
  }
  throw_unreachable("greater_than");
}

ast::value *dispatch::greater_equal(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float >= float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fcmpOGE(input, other), scalar_ty);
  // int >= int
  else if (scalar_ty->is_integer_ty()) {
    if (scalar_ty->is_integer_signed()) {
      return ctx->create_value(builder->create_icmpSGE(input, other), scalar_ty);
    } else {
      return ctx->create_value(builder->create_icmpUGE(input, other), scalar_ty);
    }
  }
  throw_unreachable("greater_equal");
}

ast::value *dispatch::less_than(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float < float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fcmpOLT(input, other), scalar_ty);
  // int < int
  else if (scalar_ty->is_integer_ty()) {
    if (scalar_ty->is_integer_signed())
      return ctx->create_value(builder->create_icmpSLT(input, other), scalar_ty);
    else
      return ctx->create_value(builder->create_icmpULT(input, other), scalar_ty);
  }
  throw_unreachable("less_than");
}

ast::value *dispatch::less_equal(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float < float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fcmpOLE(input, other), scalar_ty);
  // int < int
  else if (scalar_ty->is_integer_ty()) {
    if (scalar_ty->is_integer_signed())
      return ctx->create_value(builder->create_icmpSLE(input, other), scalar_ty);
    else
      return ctx->create_value(builder->create_icmpULE(input, other), scalar_ty);
  }
  throw_unreachable("less_equal");
}

ast::value *dispatch::equal(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float == float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fcmpOEQ(input, other), scalar_ty);
  // int == int
  else if (scalar_ty->is_integer_ty())
    return ctx->create_value(builder->create_icmpEQ(input, other), scalar_ty);
  throw_unreachable("equal");
}

ast::value *dispatch::not_equal(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder) {
  binary_op_type_checking(input, other, builder);
  ast::type *scalar_ty = input->get_type()->get_scalar_ty();
  // float == float
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_fcmpUNE(input, other), scalar_ty);
  // int == int
  else if (scalar_ty->is_integer_ty())
    return ctx->create_value(builder->create_icmpNE(input, other), scalar_ty);
  throw_unreachable("equal");
}

//===----------------------------------------------------------------------===//
//                               Block Creation
//===----------------------------------------------------------------------===//

// create_ctx_and_ir(ctx, builder, ...);

ast::value* dispatch::arange(int start, int end, ast::context *ctx, ir::builder *builder) {
  return ctx->create_value(builder->get_range(start, end), /*ctx ty*/);
}

ast::value* dispatch::zeros(shape_t shape, ast::type *dtype, ast::context *ctx, ir::builder *builder) {
  // TODO: ctx constant here
  ir::value *_0 = ir::constant::get_null_value(dtype);
  return ctx->create_value(builder->create_splat(_0, shape), dtype);
}

//===----------------------------------------------------------------------===//
//                               Shape Manipulation
//===----------------------------------------------------------------------===//


ast::value *dispatch::reshape(ast::value *input, shape_t dst_shape, ast::context *ctx, ir::builder *builder) {
  unsigned numel = 1;
  for(unsigned s: dst_shape) numel *= s;
  if(input->get_type()->get_tile_num_elements() != numel)
    throw semantic_error("cannot reshape block of different shape");
  return ctx->create_value(builder->create_reshape(input, dst_shape), /*ty*/);
}

ast::value *dispatch::cat(ast::value *lhs, ast::value *rhs, ast::context *ctx, ir::builder *builder) {
  return ctx->create_value(builder->create_cat(lhs, rhs), /*ty*/);
}

ast::value *dispatch::broadcast(ast::value *input, shape_t shape, ast::context *ctx, ir::builder *builder) {
  if (!input->get_type()->is_block_ty())
    return ctx->create_value(builder->create_splat(input, shape), /*ty*/);
  auto src_shape = input->get_type()->get_block_shapes();
  if (src_shape.size() != shape.size())
    throw std::runtime_error("Cannot broadcast");
  if(shape == src_shape)
    return input;
  return ctx->create_value(builder->create_broadcast(input, shape), /*ty*/);
}

std::tuple<ast::value*, ast::value*> dispatch::broadcast(ast::value *lhs, ast::value* rhs, ast::context *ctx, ir::builder *builder) {
  ast::type *lhs_ty = lhs->get_type();
  ast::type *rhs_ty = rhs->get_type();

  // make_shape_compatible(block, scalar)
  if (lhs_ty->is_block_ty() && !rhs_ty->is_block_ty())
    rhs = ctx->create_value(builder->create_splat(rhs, lhs_ty->get_block_shapes()), rhs_ty);
  // make_shape_compatible(scalar, block)
  else if (!lhs_ty->is_block_ty() && rhs_ty->is_block_ty())
    lhs = ctx->create_value(builder->create_splat(lhs, rhs_ty->get_block_shapes()), lhs_ty);
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
      lhs = ctx->create_value(builder->create_broadcast(lhs, ret_shape), lhs_ty);
    if (rhs_shape != ret_shape)
      rhs = ctx->create_value(builder->create_broadcast(rhs, ret_shape), rhs_ty);
  }
  return std::make_tuple(lhs, rhs);
}

ast::value *dispatch::bitcast(ast::value *input, ast::type *dst_ty, ast::context *ctx, ir::builder *builder){
  ast::type *src_ty = input->get_type();
  if (src_ty->is_block_ty())
    dst_ty = ir::block_type::get(dst_ty, input->get_type()->get_block_shapes());
  if(src_ty == dst_ty)
    return input;
  ast::type *src_sca_ty = src_ty->get_scalar_ty();
  ast::type *dst_sca_ty = dst_ty->get_scalar_ty();
  if(src_sca_ty->is_pointer_ty() || dst_sca_ty->is_pointer_ty())
    return ctx->create_value(cast(input, dst_ty, builder), dst_sca_ty);
  // Bitcast
  int src_bits = src_sca_ty->get_primitive_size_in_bits();
  int dst_bits = dst_sca_ty->get_primitive_size_in_bits();
  if(src_bits != dst_bits)
    throw std::runtime_error("Cannot bitcast data-type of size " + std::to_string(src_bits) +
                             "to data-type of size " + std::to_string(dst_bits));
  return ctx->create_value(builder->create_cast(ir::BitCast, input, dst_ty), dst_ty);
}

ast::value *dispatch::cast(ast::value *input, ast::type *dst_ty, ast::context *ctx, ir::builder *builder) {
  ast::type *src_ty = input->get_type();
  if (src_ty->is_block_ty())
    dst_ty = ast::block_type::get(dst_ty, input->get_type()->get_block_shapes());
  if (src_ty == dst_ty)
    return input;
  ast::type *src_sca_ty = src_ty->get_scalar_ty();
  ast::type *dst_sca_ty = dst_ty->get_scalar_ty();
  // FP Truncation
  bool truncate_fp = src_sca_ty->is_floating_point_ty() &&
                     dst_sca_ty->is_floating_point_ty() &&
                     src_sca_ty->get_fp_mantissa_width() > dst_sca_ty->get_fp_mantissa_width();
  if (truncate_fp)
    return ctx->create_value(builder->create_fp_trunc(input, dst_ty), dst_ty);
  // FP Extension
  bool ext_fp = src_sca_ty->is_floating_point_ty() &&
                dst_sca_ty->is_floating_point_ty() &&
                src_sca_ty->get_fp_mantissa_width() < dst_sca_ty->get_fp_mantissa_width();
  if (ext_fp)
    return ctx->create_value(builder->create_fp_ext(input, dst_ty), dst_ty);
  // Int cast
  if (src_sca_ty->is_integer_ty() && dst_sca_ty->is_integer_ty() &&
      (src_sca_ty->get_integer_bitwidth() != dst_sca_ty->get_integer_bitwidth() ||
       src_sca_ty->get_integer_signedness() != dst_sca_ty->get_integer_signedness())) {
    bool sign_extend = src_sca_ty->is_integer_signed() && src_sca_ty != builder->get_int1_ty();
    return ctx->create_value(builder->create_int_cast(input, dst_ty, sign_extend), dst_ty);
  }
  // Float -> Int
  if (src_sca_ty->is_floating_point_ty() && dst_sca_ty->is_integer_ty()){
    if(dst_sca_ty->is_bool_ty())
      return ctx->create_value(builder->create_fp_to_ui(input, dst_ty), dst_ty);
    else
      return ctx->create_value(builder->create_fp_to_si(input, dst_ty), dst_ty);
  }
  // int -> Float
  if (src_sca_ty->is_integer_ty() && dst_sca_ty->is_floating_point_ty()){
    if (src_sca_ty->is_bool_ty() || !src_sca_ty->is_integer_signed())
      return ctx->create_value(builder->create_ui_to_fp(input, dst_ty), dst_ty);
    else
      return ctx->create_value(builder->create_si_to_fp(input, dst_ty), dst_ty);
  }
  if (src_sca_ty->is_pointer_ty() && dst_sca_ty->is_integer_ty()){
    int bitwidth = dst_sca_ty->get_integer_bitwidth();
    if(bitwidth == 64)
      return ctx->create_value(builder->create_cast(ir::PtrToInt, input, dst_ty), dst_ty);
    if(bitwidth == 1)
      return dispatch::not_equal(dispatch::cast(input, builder->get_int64_ty(), builder),
                                 builder->get_int64(0),
                                 ctx,
                                 builder);
  }
  if (!src_sca_ty->is_pointer_ty() && dst_sca_ty->is_pointer_ty())
    return ctx->create_value(builder->create_cast(ir::IntToPtr, input, dst_ty), /*ty*/);
  // Ptr -> Ptr
  if (src_sca_ty->is_pointer_ty() && dst_sca_ty->is_pointer_ty())
    return ctx->create_value(builder->create_cast(ir::BitCast, input, dst_ty), /*ty*/);
  // * -> Bool
  if (dst_sca_ty->is_bool_ty()) {
    if (src_sca_ty->is_pointer_ty())
      input = cast(input, builder->get_int64_ty(), ctx, builder);
    ir::value *other = builder->get_int64(0);
    if (src_ty->is_bool_ty())
      other = ctx->create_value(builder->create_splat(other, src_ty->get_block_shapes()), /*ty*/);
    return ctx->create_value(builder->create_icmpNE(input, other), /*ty*/);
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
  if(elt_ty == builder->get_int1_ty()){
    elt_ty = builder->get_int8_ty();
    ptr_ty = pointer_type::get(elt_ty, ptr_ty->get_pointer_address_space());
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
    return ctx->create_value(builder->create_load(ptr, cache, is_volatile), /*ty*/);
  if (!mask)
    throw std::runtime_error("`other` cannot be provided without `mask`");
  auto shape = ptr->get_type()->get_block_shapes();
  if(!other){
    other = ir::undef_value::get(elt_ty);
    if(ptr->get_type()->is_block_ty())
      other = builder->create_splat(other, ptr->get_type()->get_block_shapes());
  }
  return builder->create_masked_load(ptr, mask, other, cache, is_volatile);
}

ir::value *dispatch::store(ir::value* ptr, ir::value *val, ir::value* mask, ir::builder *builder) {
  if(!ptr->get_type()->get_scalar_ty()->is_pointer_ty())
    throw semantic_error("Pointer argument of store instruction is " + ptr->get_type()->repr());
  if(ptr->get_type()->is_block_ty())
    val = dispatch::broadcast(val, ptr->get_type()->get_block_shapes(), builder);
  if(mask)
    mask = dispatch::broadcast(mask, ptr->get_type()->get_block_shapes(), builder);
  ir::type *ptr_ty = ptr->get_type()->get_scalar_ty();
  ir::type *elt_ty = ptr_ty->get_pointer_element_ty();
  // treat bool* as int8*
  if(elt_ty == builder->get_int1_ty()){
    elt_ty = builder->get_int8_ty();
    ptr_ty = pointer_type::get(elt_ty, ptr_ty->get_pointer_address_space());
    ptr = dispatch::cast(ptr, ptr_ty, builder);
  }
  // cast to target data-type
  val = dispatch::cast(val, elt_ty, builder);
  if (!mask)
    return builder->create_store(ptr, val);
  if(!mask->get_type()->get_scalar_ty()->is_bool_ty())
    throw semantic_error("Mask must have boolean scalar type");
  return builder->create_masked_store(ptr, val, mask);
}

ir::value *dispatch::atomic_cas(ir::value* ptr, ir::value *cmp, ir::value *val, ir::builder *builder){
  return builder->create_atomic_cas(ptr, cmp, val);
}

static void atom_red_typechecking(ir::value*& ptr, ir::value *&val, ir::value *&mask, ir::builder *builder){
  if(!ptr->get_type()->get_scalar_ty()->is_pointer_ty())
    throw semantic_error("Pointer argument of store instruction is " + ptr->get_type()->repr());
  if(ptr->get_type()->is_block_ty()){
    if(mask){
      mask = dispatch::broadcast(mask, ptr->get_type()->get_block_shapes(), builder);
    }
    if(val){
      val = dispatch::broadcast(val, ptr->get_type()->get_block_shapes(), builder);
    }
  }
  val = dispatch::cast(val, ptr->get_type()->get_scalar_ty()->get_pointer_element_ty(), builder);
  if(!mask){
    mask = builder->get_int1(true);
    if(ptr->get_type()->is_block_ty())
      mask = builder->create_splat(mask, ptr->get_type()->get_block_shapes());
  }
}

ir::value *dispatch::atomic_max(ir::value* ptr, ir::value *val, ir::value *mask, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, builder);
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
  return where(pos, pos_ret, neg_ret, builder);
}

ir::value *dispatch::atomic_min(ir::value* ptr, ir::value *val, ir::value *mask, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, builder);
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

ir::value *dispatch::atomic_add(ir::value* ptr, ir::value *val, ir::value *mask, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, builder);
  ir::type* sca_ty = val->get_type()->get_scalar_ty();
  auto op = sca_ty->is_floating_point_ty() ? ir::atomic_rmw_op_t::FAdd : ir::atomic_rmw_op_t::Add;
  return builder->create_atomic_rmw(op, ptr, val, mask);
}

ir::value *dispatch::atomic_and(ir::value* ptr, ir::value *val, ir::value *mask, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, builder);
  return builder->create_atomic_rmw(ir::atomic_rmw_op_t::And, ptr, val, mask);
}

ir::value *dispatch::atomic_or(ir::value* ptr, ir::value *val, ir::value *mask, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, builder);
  return builder->create_atomic_rmw(ir::atomic_rmw_op_t::Or, ptr, val, mask);
}

ir::value *dispatch::atomic_xor(ir::value* ptr, ir::value *val, ir::value *mask, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, builder);
  return builder->create_atomic_rmw(ir::atomic_rmw_op_t::Xor, ptr, val, mask);
}

ir::value *dispatch::atomic_xchg(ir::value* ptr, ir::value *val, ir::value *mask, ir::builder *builder){
  atom_red_typechecking(ptr, val, mask, builder);
  ir::type* sca_ty = val->get_type()->get_scalar_ty();
  return builder->create_atomic_rmw(ir::atomic_rmw_op_t::Xchg, ptr, val, mask);
}

//===----------------------------------------------------------------------===//
//                               Linear Algebra
//===----------------------------------------------------------------------===//

ast::value *dispatch::dot(ast::value *lhs, ast::value *rhs, ast::constant_int *allow_tf32, ast::context *ctx, ir::builder *builder) {
  ast::value *_0 = nullptr;
  if (lhs->get_type()->is_int_or_tileint_ty())
    _0 = ctx->create_value(builder->get_int32(0), ctx->get_int32_ty());
  else
    _0 = ctx->create_value(builder->get_float32(0), ctx->get_float32_ty());
  unsigned M = lhs->get_type()->get_block_shapes()[0];
  unsigned N = rhs->get_type()->get_block_shapes()[1];
  _0 = ctx->create_value(builder->create_splat(_0, {M, N}), /*ty*/);
  bool _allow_tf32 = allow_tf32->get_value() != 0;
  return ctx->create_value(builder->create_dot(lhs, rhs, _0, _allow_tf32), /*ty*/);
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
  return builder->create_select(condition, x, y);
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
    input = dispatch::cast(input, type::get_int32_ty(scalar_ty->get_context()), ctx, builder);
  if (scalar_ty->is_floating_point_ty())
    return ctx->create_value(builder->create_reduce(input, FLOAT_OP, axis), /*ty*/);
  else if (scalar_ty->is_integer_ty())
    return ctx->create_value(builder->create_reduce(input, INT_OP, axis), /*ty*/);
  throw_unreachable(name);
}

ast::value *dispatch::min(ast::value *input, unsigned int axis, ast::context *ctx, ir::builder *builder) {
  return reduce_impl(input, axis, ctx, builder, "min", ir::reduce_inst::FMIN, ir::reduce_inst::MIN);
}

ast::value *dispatch::max(ast::value *input, unsigned int axis, ast::context *ctx, ir::builder *builder) {
  return reduce_impl(input, axis, ctx, builder, "max", ir::reduce_inst::FMAX, ir::reduce_inst::MAX);
}

ast::value *dispatch::ast(ast::value *input, unsigned int axis, ast::context *ctx, ir::builder *builder) {
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
  binary_op_type_checking(x, y, builder);
  return ctx->create_value(builder->insert(umulhi_inst::create(x, y)), x->get_type());
}

ast::value *dispatch::exp(ast::value *x, ast::context *ctx, ir::builder *builder) {
  return ctx->create_value(builder->create_exp(x), x->get_type());
}

ast::value *dispatch::log(ast::value *x, ast::context *ctx, ir::builder *builder) {
  return ctx->create_value(builder->create_log(x), x->get_type());
}

ast::value *dispatch::cos(ast::value *x, ast::context *ctx, ir::builder *builder) {
  return ctx->create_value(builder->create_cos(x), x->get_type());
}

ast::value *dispatch::sin(ast::value *x, ast::context *ctx, ir::builder *builder) {
  return ctx->create_value(builder->create_sin(x), x->get_type());
}

ast::value *dispatch::sqrt(ast::value *x, ast::context *ctx, ir::builder *builder) {
  return ctx->create_value(builder->create_sqrt(x), x->get_type());
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
  return ctx->create_value(builder->create_barrier(), /*void?*/);
}


}
}
