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
 * \file mkldnn_quantized_flatten.cc
 * \brief
 */

#if MXNET_USE_MKLDNN == 1

#include "../../tensor/matrix_op-inl.h"
#include "../../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../nn/mkldnn/mkldnn_reshape-inl.h"
#include "../quantization_utils.h"

namespace mxnet {
namespace op {

class MKLDNNQuantizedFlattenFwd : public MKLDNNReshapeFwd {
public:
  explicit MKLDNNQuantizedFlattenFwd(const OpReqType &req,
                                     const NDArray &input,
                                     const NDArray &output)
           : MKLDNNReshapeFwd(req, input, output) {}
};

static MKLDNNQuantizedFlattenFwd &GetQuantizedFlattenForward(const OpReqType &req,
                                                             const NDArray &input,
                                                             const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature,
                                         MKLDNNQuantizedFlattenFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature,
                                            MKLDNNQuantizedFlattenFwd, OpHash> fwds;
#endif
  OpSignature key;
  key.AddSign(input);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNQuantizedFlattenFwd fwd(req, input, output);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNQuantizedFlattenForward(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs) {
  if (req[0] == kNullOp) return;
  CHECK_NE(req[0], kAddTo) << "kAddTo is not supported yet";

  outputs[1].data().dptr<float>()[0] = inputs[1].data().dptr<float>()[0];
  outputs[2].data().dptr<float>()[0] = inputs[2].data().dptr<float>()[0];

  auto fwd = GetQuantizedFlattenForward(req[0], inputs[0], outputs[0]);
  auto ws_size = fwd.GetWorkspaceSize();
  void* ws_ptr = nullptr;
  if (ws_size) {
    mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
    mshadow::Tensor<cpu, 1, char> ws = ctx.requested[0]
      .get_space_typed<cpu, 1, char>(mshadow::Shape1(ws_size), s);
    ws_ptr = reinterpret_cast<void*>(ws.dptr_);
  }

  fwd.Execute(inputs[0], outputs[0], ws_ptr);
}

NNVM_REGISTER_OP(_contrib_quantized_flatten)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedFlattenForward);

}
}

#endif
