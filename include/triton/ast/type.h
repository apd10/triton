#ifndef _TRITON_AST_TYPE_H_
#define _TRITON_AST_TYPE_H_

//===----------------------------------------------------------------------===//
//
/// \file
/// Frontend types. Will be lowered to IR types by dispatch.cc
//
//===----------------------------------------------------------------------===//


namespace triton {
namespace ast {

enum class signedness { SIGNED, UNSIGNED };

class type {
public:
  enum id_t : unsigned {
    VoidTyID = 0,
    // fp ty
    FP8TyID,
    FP16TyID,
    BF16TyID,
    FP32TyID,
    FP64TyID,
    // int ty
    INT8TyID,
    INT16TyID,
    INT32TyID,
    INT64TyID,
    UINT8TyID,
    UINT16TyID,
    UINT32TyID,
    UINT64TyID,
    // other
    FunctionTyID,
  };
private:
  id_t id_;

public:
  bool is_signed_integer() const;
  bool is_unsigned_integer() const;

public:
  // Factory methods
  static type *get_void_ty();
  static type *get_label_ty();
  static type *get_fp8_ty();
  static type *get_fp16_ty();
  static type *get_bf16_ty();
  static type *get_fp32_ty();
  static type *get_fp64_ty();
  // int
  static type *get_int1_ty();
  static type *get_int8_ty();
  static type *get_int16_ty();
  static type *get_int32_ty();
  static type *get_int64_ty();
  static type *get_int128_ty();
  // uint
  static type *get_uint8_ty();
  static type *get_uint16_ty();
  static type *get_uint32_ty();
  static type *get_uint64_ty();
  static type *get_uint128_ty();
};

} // namespace ast
} // namespace triton


#endif // _TRITON_AST_TYPE_H_
