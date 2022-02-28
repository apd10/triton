#pragma once

#ifndef _TRITON_IR_DISPATCH_H_
#define _TRITON_IR_DISPATCH_H_

#include "triton/ir/builder.h"
#include "triton/ast/ast.h"
#include <stdexcept>

namespace triton{
namespace ir{


/*----------------------------------------------
 higher level functions that follow the likely
 semantics of most expected frontends
 ----------------------------------------------*/

struct semantic_error: public std::runtime_error {
  semantic_error(const std::string& msg):
    std::runtime_error(msg) { }
};


// Since functions in dispatch.cc do semantic anlysis (at ast level) and ir codegen,
// we will need both ast::context & ir::builder.
// Note that ast::value encapsulates ir::value
struct dispatch{
  typedef ir::type::block_shapes_t shape_t;


  // programming model
  static ast::value *program_id(int axis, ast::context *ctx, ir::builder *builder);
  static ast::value *num_programs(int axis, ast::context *ctx, ir::builder *builder);

  // binary operators
  static ast::value *add(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *sub(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *mul(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *truediv(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *floordiv(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *fdiv(ast::value *input, ast::value *other, ir::constant_int* ieee_rounding, ast::context *ctx, ir::builder *builder);
  static ast::value *mod(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *and_(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *or_(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *xor_(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *lshr(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *shl(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);

  // unary operators
  static ast::value *plus(ast::value *input, ast::context *ctx, ir::builder *builder);
  static ast::value *minus(ast::value *input, ast::context *ctx, ir::builder *builder);
  static ast::value *invert(ast::value *input, ast::context *ctx, ir::builder *builder);

  // comparison operators
  static ast::value *greater_than(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *greater_equal(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *less_than(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *less_equal(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *equal(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);
  static ast::value *not_equal(ast::value *input, ast::value *other, ast::context *ctx, ir::builder *builder);

  // block creation
  static ast::value* arange(int start, int end, ast::context *ctx, ir::builder *builder);
  static ast::value* zeros(shape_t shape, ast::type *dtype, ast::context *ctx, ir::builder *builder);


  // casting ops
  static ast::value *reshape(ast::value *input, shape_t shape, ast::context *ctx, ir::builder *builder);
  static ast::value *cat(ast::value *lhs, ast::value *rhs, ast::context *ctx, ir::builder *builder);
  static ast::value *broadcast(ast::value *input, shape_t shape, ast::context *ctx, ir::builder *builder);
  static std::tuple<ast::value*, ast::value*> broadcast(ast::value *lhs, ast::value* rhs, ast::context *ctx, ir::builder *builder);
  static ast::value *bitcast(ast::value *input, ast::type *type, ast::context *ctx, ir::builder *builder);
  static ast::value *cast(ast::value *input, ast::type *type, ast::context *ctx, ir::builder *builder);

  // memory operators
  static ast::value *load(ast::value* ptr, ast::value* mask, ast::value* other, const std::string &cache, int is_volatile, ast::context *ctx, ir::builder *builder);
  static ast::value *store(ast::value* ptr, ast::value *value, ast::value *mask, ast::context *ctx, ir::builder *builder);
  static ast::value *atomic_cas(ast::value* ptr, ast::value *cmp, ast::value *val, ast::context *ctx, ir::builder *builder);
  static ast::value *atomic_add(ast::value* ptr, ast::value *val, ast::value *msk, ast::context *ctx, ir::builder *builder);
  static ast::value *atomic_max(ast::value* ptr, ast::value *val, ast::value *msk, ast::context *ctx, ir::builder *builder);
  static ast::value *atomic_min(ast::value* ptr, ast::value *val, ast::value *msk, ast::context *ctx, ir::builder *builder);
  static ast::value *atomic_and(ast::value* ptr, ast::value *val, ast::value *msk, ast::context *ctx, ir::builder *builder);
  static ast::value *atomic_or(ast::value* ptr, ast::value *val, ast::value *msk, ast::context *ctx, ir::builder *builder);
  static ast::value *atomic_xor(ast::value* ptr, ast::value *val, ast::value *msk, ast::context *ctx, ir::builder *builder);
  static ast::value *atomic_xchg(ast::value* ptr, ast::value *val, ast::value *msk, ast::context *ctx, ir::builder *builder);

  // linear algebra
  static ast::value *dot(ast::value *lhs, ast::value *rhs, ir::constant_int *allow_tf32, ast::context *ctx, ir::builder *builder);

  // indexing
  static ast::value *where(ast::value* condition, ast::value *x, ast::value *y, ast::context *ctx, ir::builder *builder);

  // reduction
  static ast::value *min(ast::value *input, unsigned int axis, ast::context *ctx, ir::builder *builder);
  static ast::value *max(ast::value *input, unsigned int axis, ast::context *ctx, ir::builder *builder);
  static ast::value *sum(ast::value *input, unsigned int axis, ast::context *ctx, ir::builder *builder);
  static ast::value *xor_sum(ast::value *input, unsigned int axis, ast::context *ctx, ir::builder *builder);

  // math
  static ast::value *umulhi(ast::value *x, ast::value *y, ast::context *ctx, ir::builder *builder);
  static ast::value *exp(ast::value *x, ast::context *ctx, ir::builder *builder);
  static ast::value *log(ast::value *x, ast::context *ctx, ir::builder *builder);
  static ast::value *cos(ast::value *x, ast::context *ctx, ir::builder *builder);
  static ast::value *sin(ast::value *x, ast::context *ctx, ir::builder *builder);
  static ast::value *sqrt(ast::value *x, ast::context *ctx, ir::builder *builder);

  // internal (debug/optimization)
  static ast::value *multiple_of(ast::value *x, int value, ast::context *ctx, ir::builder *builder);
  static ast::value *max_contiguous(ast::value *x, int value, ast::context *ctx, ir::builder *builder);
  static ast::value *debug_barrier(ast::context *ctx, ir::builder *builder);

  // TODO: rules for ast::type => ir::type lowering
};

}
}

#endif
