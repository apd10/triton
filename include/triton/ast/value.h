#ifndef _TRITON_AST_VALUE_H_
#define _TRITON_AST_VALUE_H_

#include "triton/ast/type.h"

namespace triton {
namespace ast {

/// Represent value at the ast level. It encapsulates generated ir value.
class value {
  ir::value *ir_value_;
  /// Frontend type
  type *type_;

public:
  value(ir::value *v, type *ast_ty) : ir_value_(v), type_(ast_ty) {}

  ir::value *get_ir_value() const { return ir_value_; }
  type *get_type() const { return type_; }
};

}
}

#endif
