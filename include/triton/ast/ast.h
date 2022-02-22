#ifndef _TRITON_FRONTEND_AST_H_
#define _TRITON_FRONTEND_AST_H_

namespace triton {
namespace ast {

class context_impl {

};

class context {

};


class value {
  /// Generated ir value
  ir::value *ir_value_;
  /// Frontend type
  type *type_;

public:
  value(ir::value *v) : ir_value_(v) {}
};

} // namespace ast
} // namespace triton

#endif
