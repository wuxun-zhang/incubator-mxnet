/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file np_matrix_op.cc
 * \brief CPU Implementation of numpy matrix operations
 */

#include <vector>
#include "./np_matrix_op-inl.h"
#include "../nn/concat-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyTransposeParam);
DMLC_REGISTER_PARAMETER(NumpyRollParam);

bool NumpyTransposeShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& shp = (*in_attrs)[0];
  CHECK_LE(shp.ndim(), 6) << "Transpose support at most 6 dimensions";
  mxnet::TShape ret(shp.ndim(), -1);
  if (ndim_is_known(param.axes)) {
    CHECK_EQ(shp.ndim(), param.axes.ndim());
    for (int i = 0; i < shp.ndim(); ++i) {
      CHECK(param.axes[i] < static_cast<int64_t>(shp.ndim()));
      ret[i] = shp[param.axes[i]];
    }
  } else {
    for (int i = 0; i < shp.ndim(); ++i) {
      ret[i] = shp[shp.ndim()-1-i];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return shape_is_known(ret);
}

NNVM_REGISTER_OP(_np_transpose)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyTransposeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyTransposeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(n->attrs.parsed);
    if (ndim_is_known(param.axes)) {
      mxnet::TShape axes = mxnet::TShape(param.axes.ndim(), -1);
      for (int i = 0; i < axes.ndim(); ++i) {
        axes[param.axes[i]] = i;
      }
      std::ostringstream os;
      os << axes;
      return MakeNonlossGradNode("transpose", n, ograds, {}, {{"axes", os.str()}});
    } else {
      return MakeNonlossGradNode("transpose", n, ograds, {},
                                 std::unordered_map<std::string, std::string>());
    }
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyTranspose<cpu>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.add_argument("a", "NDArray-or-Symbol", "Source input")
.add_arguments(NumpyTransposeParam::__FIELDS__());

struct NumpyReshapeParam : public dmlc::Parameter<NumpyReshapeParam> {
  mxnet::TShape newshape;
  std::string order;
  DMLC_DECLARE_PARAMETER(NumpyReshapeParam) {
      DMLC_DECLARE_FIELD(newshape)
          .describe("The new shape should be compatible with the original shape."
                    " If an integer, then the result will be a 1-D array of that length."
                    " One shape dimension can be -1. In this case, the value is inferred"
                    " from the length of the array and remaining dimensions.");
      DMLC_DECLARE_FIELD(order)
      .set_default("C")
      .describe("Read the elements of a using this index order, and place the elements into"
                " the reshaped array using this index order. 'C' means to read/write the elements"
                " using C-like index order, with the last axis index changing fastest, back to the"
                " first axis index changing slowest. Note that currently only C-like order is"
                " supported");
  }
};

DMLC_REGISTER_PARAMETER(NumpyReshapeParam);

bool NumpyReshapeInferShape(const mxnet::TShape& src, mxnet::TShape* dst) {
  if (shape_is_known(src) && shape_is_known(*dst)) {
    CHECK_EQ(src.Size(), dst->Size()) << "Cannot reshape array of size "
                                      << src.Size() << " into shape " << *dst;
    return true;
  } else if (!shape_is_known(src) || !ndim_is_known(*dst)) {
    return false;
  } else {
    int unknown_axis = -1;
    dim_t known_dim_size_prod = 1;
    for (int i = 0; i < dst->ndim(); ++i) {
      if (!dim_size_is_known(*dst, i)) {
        if (unknown_axis == -1) {
          unknown_axis = i;
        } else {
          return false;  // more than one unknown dim
        }
      } else {
        known_dim_size_prod *= (*dst)[i];
      }
    }
    CHECK_NE(known_dim_size_prod, 0) << "Cannot reshape array of size "
                                     << src.Size() << " into shape " << *dst;
    CHECK_EQ(src.Size() % known_dim_size_prod, 0) << "Cannot reshape array of size "
                                                  << src.Size() << " into shape " << *dst;
    (*dst)[unknown_axis] = src.Size() / known_dim_size_prod;
    return true;
  }
}

bool NumpyReshapeShape(const nnvm::NodeAttrs& attrs,
                       mxnet::ShapeVector* in_attrs,
                       mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyReshapeParam& param = nnvm::get<NumpyReshapeParam>(attrs.parsed);
  // sanity check
  bool has_unknown_dim_size = false;
  for (int i = 0; i < param.newshape.ndim(); ++i) {
    if (param.newshape[i] < 0) {
      CHECK_EQ(param.newshape[i], -1) << "The shape dimension size to inferred must be -1";
      CHECK(!has_unknown_dim_size) << "Can only specify one unknown dimension";
      has_unknown_dim_size = true;
    }
  }

  mxnet::TShape target_shape = param.newshape;
  bool success = NumpyReshapeInferShape(in_attrs->at(0), &target_shape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, target_shape);
  if (!success) {
    success = NumpyReshapeInferShape(out_attrs->at(0), &in_attrs->at(0));
  }
  return success;
}

NNVM_REGISTER_OP(_np_reshape)
.describe(R"code()code" ADD_FILELINE)
.add_alias("_npi_reshape")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReshapeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyReshapeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_reshape"})
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.add_argument("a", "NDArray-or-Symbol", "Array to be reshaped.")
.add_arguments(NumpyReshapeParam::__FIELDS__());

bool NumpySqueezeShape(const nnvm::NodeAttrs& attrs,
                       mxnet::ShapeVector *in_attrs,
                       mxnet::ShapeVector *out_attrs) {
  const SqueezeParam& param = nnvm::get<SqueezeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [a]";
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& dshape = in_attrs->at(0);
  const int dndim = dshape.ndim();
  if (!shape_is_known(dshape)) return false;
  mxnet::TShape oshape = dshape;
  // special case, scalar tensor
  if (dshape.ndim() == 0) {
    if (param.axis.has_value()) {
      mxnet::Tuple<int> axes = param.axis.value();
      CHECK_EQ(axes.ndim(), 1) << "cannot specify more than one axis for a scalar tensor";
      CHECK(axes[0] == 0 || axes[0] == -1) << "axis " << axes[0]
                                           << " is out of bounds of array of dimension 0";
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, -1));
    return true;
  }
  if (param.axis.has_value()) {
    // preprocess axis
    mxnet::Tuple<int> axes = param.axis.value();
    for (int i = 0; i < axes.ndim(); ++i) {
      if (axes[i] < 0) {
        axes[i] += dndim;
        CHECK_GE(axes[i], 0)
            << "axis " << axes[i] - dndim << " is out of bounds for array of dimension " << dndim;
      }
      CHECK_LT(axes[i], dndim)
          << "axis " << axes[i] << " is out of bounds for array of dimension " << dndim;
      CHECK_EQ(dshape[axes[i]], 1)
          << "cannot select an axis to squeeze out which has size="
          << dshape[axes[i]] << " not equal to one";
      CHECK_NE(oshape[axes[i]], 0) << "duplicate value in axis";
      oshape[axes[i]] = -1;
    }
  } else {
    for (int i = 0; i < oshape.ndim(); ++i) {
      if (oshape[i] == 1) oshape[i] = -1;
    }
  }
  size_t oshape_size = SqueezeShapeHelper(&oshape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(oshape.data(), oshape.data()+oshape_size));
  return true;
}

NNVM_REGISTER_OP(_np_squeeze)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SqueezeParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", NumpySqueezeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_squeeze"})
.add_argument("a", "NDArray-or-Symbol", "data to squeeze")
.add_arguments(SqueezeParam::__FIELDS__());

bool ConcatShape(const nnvm::NodeAttrs& attrs,
                 mxnet::ShapeVector *in_shape,
                 mxnet::ShapeVector *out_shape);

bool ConcatType(const nnvm::NodeAttrs& attrs,
                std::vector<int> *in_type,
                std::vector<int> *out_type);

struct NumpyConcatGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    CHECK_EQ(ograds.size(), 1);
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};


NNVM_REGISTER_OP(_npi_concatenate)
.describe(R"code(Join a sequence of arrays along an existing axis.)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  return params.num_args;
})
.set_num_outputs(1)
.set_attr_parser(ParamParser<ConcatParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
    std::vector<std::string> ret;
    for (int i = 0; i < params.num_args; ++i) {
      ret.push_back(std::string("data") + std::to_string(i));
    }
    return ret;
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"out"};
})
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<nnvm::FInferType>("FInferType", ConcatType)
.set_attr<mxnet::FInferShape>("FInferShape", ConcatShape)
.set_attr<FCompute>("FCompute<cpu>", ConcatCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", NumpyConcatGrad{"_backward_np_concat"})
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
.add_arguments(ConcatParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_concat)
.set_num_outputs([](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  return params.num_args;
})
.set_attr_parser(ParamParser<ConcatParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", ConcatGradCompute<cpu>);

NNVM_REGISTER_OP(_npi_stack)
.describe(R"code(Join a sequence of arrays along a new axis.

The axis parameter specifies the index of the new axis in the dimensions of the
result. For example, if axis=0 it will be the first dimension and if axis=-1 it
will be the last dimension.

Examples::

  x = [1, 2]
  y = [3, 4]

  stack(x, y) = [[1, 2],
                 [3, 4]]
  stack(x, y, axis=1) = [[1, 3],
                         [2, 4]]
)code")
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const StackParam& param = dmlc::get<StackParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_args);
  })
.set_num_outputs(1)
.set_attr_parser(ParamParser<StackParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<StackParam>(attrs.parsed).num_args;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("arg") + std::to_string(i));
    }
    return ret;
  })
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<mxnet::FInferShape>("FInferShape", StackOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FCompute>("FCompute<cpu>", StackOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_stack"})
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to stack")
.add_arguments(StackParam::__FIELDS__());

bool NumpyVstackType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_type,
                     std::vector<int> *out_type) {
  const NumpyVstackParam& param = nnvm::get<NumpyVstackParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), param.num_args);
  CHECK_EQ(out_type->size(), 1);
  int dtype = -1;
  for (int i = 0; i < param.num_args; i++) {
    if (dtype == -1) {
      dtype = in_type->at(i);
    }
  }
  if (dtype == -1) {
    dtype = out_type->at(0);
  }
  for (int i = 0; i < param.num_args; i++) {
    TYPE_ASSIGN_CHECK(*in_type, i, dtype);
  }
  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  return dtype != -1;
}

bool NumpyVstackShape(const nnvm::NodeAttrs& attrs,
                      mxnet::ShapeVector* in_attrs,
                      mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyVstackParam& param = nnvm::get<NumpyVstackParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.num_args);
  std::vector<mxnet::TShape> in_attrs_tmp(param.num_args);
  TShape dshape;
  for (int i = 0; i < param.num_args; i++) {
    if ((*in_attrs)[i].ndim() == 0) {
      in_attrs_tmp[i] = TShape(2, 1);
    } else if ((*in_attrs)[i].ndim() == 1) {
      in_attrs_tmp[i] = TShape(2, 1);
      in_attrs_tmp[i][1] = (*in_attrs)[i][0];
    } else {
      in_attrs_tmp[i] = (*in_attrs)[i];
    }
    TShape tmp(in_attrs_tmp[i].ndim(), -1);
    shape_assign(&dshape, tmp);
  }
  TShape tmp((*out_attrs)[0].ndim(), -1);
  shape_assign(&dshape, tmp);
  for (int i = 0; i < param.num_args; i++) {
    SHAPE_ASSIGN_CHECK(in_attrs_tmp, i, dshape)
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape)
  if (dshape.ndim() == -1) {
    return false;
  }
  int cnt = 0, sum = 0, pos = -1;
  for (int i = 0; i < param.num_args; i++) {
    TShape tmp = in_attrs_tmp[i];
    if (!dim_size_is_known(tmp, 0)) {
      cnt++;
      pos = i;
    } else {
      sum += tmp[0];
    }
    tmp[0] = -1;
    shape_assign(&dshape, tmp);
  }
  tmp = out_attrs->at(0);
  if (!dim_size_is_known(tmp, 0)) {
    cnt++;
    pos = -1;
  } else {
    sum += tmp[0];
  }
  tmp[0] = -1;
  shape_assign(&dshape, tmp);
  for (int i = 0; i < param.num_args; i++) {
    SHAPE_ASSIGN_CHECK(in_attrs_tmp, i, dshape)
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape)\
  dshape[0] = 0;
  if (!shape_is_known(dshape)) {
    return false;
  }

  dshape[0] = sum;
  if (cnt == 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
  } else if (cnt == 1) {
    if (pos >= 0) {
      in_attrs_tmp[pos][0] = out_attrs->at(0)[0] - sum;
    } else {
      out_attrs->at(0)[0] = sum;
    }
  } else {
    return false;
  }

  for (int i = 0; i < param.num_args; i++) {
    if (in_attrs->at(i).ndim() == 1) {
      in_attrs->at(i)[0] = in_attrs_tmp[i][1];
    } else if (in_attrs->at(i).ndim() >= 2) {
      in_attrs->at(i) = in_attrs_tmp[i];
    }
  }

  return true;
}

DMLC_REGISTER_PARAMETER(NumpyVstackParam);

NNVM_REGISTER_OP(_npi_vstack)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<NumpyVstackParam>)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
  const NumpyVstackParam& param = dmlc::get<NumpyVstackParam>(attrs.parsed);
  return static_cast<uint32_t>(param.num_args);
})
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const nnvm::NodeAttrs& attrs) {
    int num_args = dmlc::get<NumpyVstackParam>(attrs.parsed).num_args;
    std::vector<std::string> ret;
    for (int i = 0; i < num_args; i++) {
      ret.push_back(std::string("arg") + std::to_string(i));
    }
    return ret;
  })
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<mxnet::FInferShape>("FInferShape", NumpyVstackShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyVstackType)
.set_attr<FCompute>("FCompute<cpu>", NumpyVstackForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_np_vstack"})
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to vstack")
.add_arguments(NumpyVstackParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_vstack)
.set_attr_parser(ParamParser<NumpyVstackParam>)
.set_num_inputs(1)
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
  const NumpyVstackParam& param = dmlc::get<NumpyVstackParam>(attrs.parsed);
  return static_cast<uint32_t>(param.num_args);
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyVstackBackward<cpu>);

inline bool NumpyRollShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_attrs,
                           mxnet::ShapeVector *out_attrs) {
  using namespace mshadow;
  const NumpyRollParam& param = nnvm::get<NumpyRollParam>(attrs.parsed);

  if (!param.shift.has_value()) {
    LOG(FATAL) << "roll missing 1 required positional argument: 'shift'.";
  }
  if (param.shift.value().ndim() > 1 &&
      param.axis.has_value() &&
      param.axis.value().ndim() != param.shift.value().ndim()) {
    LOG(FATAL) << "shift and `axis` must be a tuple of the same size.";
  }
  if (!param.axis.has_value() && param.shift.has_value() && param.shift.value().ndim() > 1) {
    LOG(FATAL) << "shift must be an int.";
  }
  if (param.axis.has_value()) {
    mxnet::TShape axes(param.axis.value());
    const index_t ndim = (*in_attrs)[0].ndim();
    for (index_t i = 0; i < axes.ndim(); i++) {
      if (axes[i] < 0) {
        axes[i] += ndim;
      }
    }
    std::sort(axes.begin(), axes.end());
    for (index_t i = 1; i < axes.ndim(); i++) {
      CHECK_LT(axes[i - 1], axes[i])
        << "axes have duplicates " << axes;
    }
    CHECK_LT(axes[axes.ndim() - 1], ndim)
      << "axis " << axes[axes.ndim() - 1]
      << " Exceeds input dimensions " << (*in_attrs)[0];
    CHECK_GE(axes[0], 0)
      << "Reduction axis " << param.axis.value()
      << " Exceeds input dimensions " << (*in_attrs)[0];
  }
  return ElemwiseShape<1, 1>(attrs, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(_np_roll)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyRollParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
     return std::vector<std::string>{"data"};
})
.set_attr<mxnet::FInferShape>("FInferShape", NumpyRollShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<mxnet::FCompute>("FCompute<cpu>", NumpyRollCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
     const NumpyRollParam& param = nnvm::get<NumpyRollParam>(n->attrs.parsed);
     if (!param.shift.has_value()) {
       LOG(FATAL) << "roll missing 1 required positional argument: 'shift'.";
     }
     mxnet::TShape shifts(param.shift.value());
     for (int i = 0; i < shifts.ndim(); ++i) {
       shifts[i] = -shifts[i];
     }
     std::ostringstream os1;
     os1 << dmlc::optional<mxnet::TShape>(shifts);
     std::ostringstream os2;
     os2 << param.axis;
     return MakeNonlossGradNode("_np_roll", n, ograds, {},
                                {{"shift", os1.str()}, {"axis", os2.str()}});
})
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& n) {
     return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NumpyRollParam::__FIELDS__());

template<>
void NumpyFlipForwardImpl<cpu>(const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<TBlob>& outputs,
                               const std::vector<index_t>& stride_,
                               const std::vector<index_t>& trailing_,
                               const index_t& flip_index) {
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mxnet_op::Kernel<reverse, cpu>::Launch(s, inputs[0].Size(), flip_index,
      inputs[0].dptr<DType>(), outputs[0].dptr<DType>(),
      stride_.data(), trailing_.data());
  });
}

DMLC_REGISTER_PARAMETER(FlipParam);

NNVM_REGISTER_OP(_npi_flip)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr_parser(ParamParser<FlipParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
  return std::vector<std::string> {"data"};
})
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest> {ResourceRequest::kTempSpace};
})
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", NumpyFlipForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_flip"})
.add_argument("data", "NDArray-or-Symbol", "Input data array")
.add_arguments(FlipParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_flip)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<FlipParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest> {ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", NumpyFlipForward<cpu>);

}  // namespace op
}  // namespace mxnet
