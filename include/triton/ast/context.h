#ifndef _TRITON_AST_CONTEXT_H_
#define _TRITON_AST_CONTEXT_H_

#include "triton/ast/value.h"
#include "triton/ast/type.h"
#include <vector>
#include <map>
#include <memory>

namespace triton {

namespace ir {
class type;
}

namespace ast {
/// Holds long-lived ast nodes (such as types, constant values)
class context {
public:
  context(ir::context &ir_ctx) : ir_ctx_(ir_ctx) {}
  context(const context &) = delete;
  context& operator=(const context &) = delete;

private:
  ir::context &ir_ctx_; ///< Each ast context is associated with an ir context
public:
  ir::context *get_ir_context() { return &ir_ctx_; }

// record ast::values
public:
  value *create_value(ir::value *, type *);
private:
  std::vector<std::unique_ptr<ast::value>> ast_values_;
  std::map<std::pair<ir::type*, type::signedness>, std::unique_ptr<type>> ast_types_;
};

} // namespace ast
} // namespace triton

#endif
