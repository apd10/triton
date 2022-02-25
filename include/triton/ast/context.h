#ifndef _TRITON_AST_CONTEXT_H_
#define _TRITON_AST_CONTEXT_H_

#include 

namespace triton {
namespace ast {
/// Holds long-lived ast nodes (such as types, constant values)
class context {
public:
  context();
  context(const context &) = delete;
  context& operator=(const context &) = delete;

public:
  type void_ty;
  // floating point types
  type fp8_ty, fp16_ty, bf16_ty, fp32_ty, fp64_ty;
  // integer types
  type int1_ty, int8_ty, int16_ty, int32_ty, int64_ty;
  type uint8_ty, uint16_ty, uint32_ty, uint64_t;
  // Pointer types
  // TODO: need pointer types?
  // Block types
  // TODO: need block types?
  
  // constants
  // std::map<std::pair<type*, uint64_t>, consta> int_constants_;
  // std::map<std::pair<type*, 

// record ast::values
public:
  value *create_value(ir::value *, );
private:
  std::vector<std::unique_ptr<ast::value>> ast_values_;
};

} // namespace ast
} // namespace triton

#endif
