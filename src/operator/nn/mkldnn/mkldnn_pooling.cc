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
 * \file mkldnn_pooling.cc
 * \brief
 * \author Tao Lv
*/

#if MXNET_USE_MKLDNN == 1

#include "./mkldnn_pooling-inl.h"

namespace mxnet {
namespace op {

void MKLDNNPoolingFwd::Init(const mxnet::NDArray &input, const mxnet::NDArray &output,
                            const mkldnn::memory::dims &kernel,
                            const mkldnn::memory::dims &strides,
                            const mkldnn::memory::dims &pad_l,
                            const mkldnn::memory::dims &pad_r,
                            const bool is_train, const mkldnn::algorithm alg_kind) {
  auto src_md = input.GetMKLDNNData()->get_desc();
  // const auto input_ndims = input.shape().ndim();
  // mkldnn::memory::dims dims;
  // if (output.shape().ndim() == 4) {
  //   dims.push_back({src_md.data.dims[0], src_md.data.dims[1],
  //                   static_cast<mkldnn_dim_t>(output.shape()[2]),
  //                   static_cast<mkldnn_dim_t>(output.shape()[3])});
  // } else {
  //   dims.push_back({src_md.data.dims[0], src_md.data.dims[1],
  //                   static_cast<mkldnn_dim_t>(output.shape()[2]),
  //                   static_cast<mkldnn_dim_t>(output.shape()[3]),
  //                   static_cast<mkldnn_dim_t>(output.shape()[4])});
  // }
  // mkldnn::memory::dims dims = {src_md.data.dims[0],
  //                              src_md.data.dims[1],
  //                              static_cast<int>(output.shape()[2]),
  //                              static_cast<int>(output.shape()[3])};
  // auto dst_md = mkldnn::memory::desc({dims},
  //                                    static_cast<mkldnn::memory::data_type>(src_md.data.data_type),
  //                                    mkldnn::memory::format_tag::any);
  auto dst_md = GetMemDesc(output);
  const mkldnn::engine engine = CpuEngine::Get()->get_engine();
  if (alg_kind != mkldnn::algorithm::pooling_max &&
      alg_kind != mkldnn::algorithm::pooling_avg &&
      alg_kind != mkldnn::algorithm::pooling_avg_include_padding &&
      alg_kind != mkldnn::algorithm::pooling_avg_exclude_padding) {
    LOG(FATAL) << "MKLDNN Pooling: algorithm is not supported";
  }

  mkldnn::prop_kind prop = mkldnn::prop_kind::forward_scoring;
  if (is_train && alg_kind != mkldnn::algorithm::pooling_avg) {
    prop = mkldnn::prop_kind::forward_training;
  }
  if (is_train && prop == mkldnn::prop_kind::forward_scoring) {
    LOG(INFO) << "MKLDNN Pooling: training with prop_kind is forward_scoring";
  }

  // mkldnn::memory::dims kernel(input_ndims - 2);
  // mkldnn::memory::dims strides(input_ndims - 2);
  // mkldnn::memory::dims pad_l(input_ndims - 2);
  // mkldnn::memory::dims pad_r(input_ndims - 2);
  // if (input_ndims == 4) {
  //   // 2d
  //   kernel[0] = in_kernel[0];
  //   kernel[1] = in_kernel[1];
  //   strides[0] = in_strides[0];
  //   strides[1] = in_strides[1];
  //   pad_l[0] = in_pad_l[0];
  //   pad_l[1] = in_pad_l[1];
  //   pad_r[0] = in_pad_r[0];
  //   pad_r[1] = in_pad_r[1];
  // } else {
  //   // 3d
  //   kernel[0] = in_kernel[0];
  //   kernel[1] = in_kernel[1];
  //   kernel[2] = in_kernel[2];
  //   strides[0] = in_strides[0];
  //   strides[1] = in_strides[1];
  //   strides[2] = in_strides[2];
  //   pad_l[0] = in_pad_l[0];
  //   pad_l[1] = in_pad_l[1];
  //   pad_l[2] = in_pad_l[2];
  //   pad_r[0] = in_pad_r[0];
  //   pad_r[1] = in_pad_r[1];
  //   pad_r[2] = in_pad_r[2];
  // }
  // const mkldnn::memory::dims strides = {stride_h,  stride_w  };
  // const mkldnn::memory::dims pad_l   = {padding_t, padding_l };
  // const mkldnn::memory::dims pad_r   = {padding_b, padding_r };
  // const mkldnn::memory::dims kernel  = {kernel_h,  kernel_w  };

  // LOG(INFO) << "Entering MKLDNNPoolingFwd::Init()";
  const auto fwd_desc = mkldnn::pooling_forward::desc(prop, alg_kind, src_md, dst_md,
                                                      strides, kernel, pad_l, pad_r);
  this->fwd_pd_.reset(new mkldnn::pooling_forward::primitive_desc(fwd_desc, engine));
  this->fwd_.reset(new mkldnn::pooling_forward(*(this->fwd_pd_)));

  // LOG(INFO) << "Exiting MKLDNNPoolingFwd::Init()";
  return;
}

void MKLDNNPoolingFwd::Execute(const NDArray &in_data,
                               const OpReqType req,
                               const NDArray& out_data,
                               const NDArray *workspace) {
  NDArray in_buffer = in_data;
  if (in_data.IsView() && in_data.IsMKLDNNData())
    in_buffer = in_data.Reorder2Default();

  auto input_mem = in_buffer.GetMKLDNNData();
  auto output_mem_t_ = CreateMKLDNNMem(out_data, this->fwd_pd_->dst_desc(), req);

  mkldnn_args_map_t args = {
    {MKLDNN_ARG_SRC, *input_mem },
    {MKLDNN_ARG_DST, *(output_mem_t_.second) },
  };

  if (this->with_workspace_) {
    auto engine = CpuEngine::Get()->get_engine();

    if (workspace == nullptr) {
        LOG(FATAL) << "MKLDNN Pooling: incorrect workspace input";
    }

    auto ws = std::make_shared<mkldnn::memory>((*(this->fwd_pd_)).workspace_desc(),
                      engine, workspace->GetMKLDNNData()->get_data_handle());
    args[MKLDNN_ARG_WORKSPACE] = *ws;
  }
  if (this->fwd_) {
    MKLDNNStream::Get()->RegisterPrimArgs(*(this->fwd_), args);
    CommitOutput(out_data, output_mem_t_);
    MKLDNNStream::Get()->Submit();
  } else {
    LOG(FATAL) << "MKLDNN Pooling: forward primitive is nullptr";
  }
}

mkldnn::algorithm GetMKLDNNPoolAlgo(const PoolingParam &param) {
  switch (param.pool_type) {
    case pool_enum::kMaxPooling:
      return mkldnn::algorithm::pooling_max;
      break;
    case pool_enum::kAvgPooling:
      if (param.count_include_pad.has_value() && !param.count_include_pad.value()) {
        return mkldnn::algorithm::pooling_avg_exclude_padding;
      } else {
        return mkldnn::algorithm::pooling_avg_include_padding;
      }
      break;
    default:
      LOG(FATAL) << "MKLDNN Pooling: Unknown pooling method.";
      return mkldnn::algorithm::pooling_max;
  }
}


void InitPoolingPrimitiveParams(const PoolingParam &param,
                                const mkldnn::memory::desc &data_md,
                                mkldnn::memory::dims &kernel,
                                mkldnn::memory::dims &strides,
                                mkldnn::memory::dims &pad_l,
                                mkldnn::memory::dims &pad_r) {
  const int kernel_ndims = kernel.size();
  if (kernel_ndims == 2) {
    CHECK_GE(param.pad.ndim(), 2);
    CHECK_GE(param.stride.ndim(), 2);
    kernel[0] = param.kernel[0];
    kernel[1] = param.kernel[1];
    pad_l[0] = param.pad[0];
    pad_l[1] = param.pad[1];
    pad_r[0] = param.pad[0];
    pad_r[1] = param.pad[1];
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];

    if (param.global_pool) {
      kernel[0] = data_md.data.dims[2];
      kernel[1] = data_md.data.dims[3];
      strides[0] = 1;
      strides[1] = 1;
    } else if (param.pooling_convention == pool_enum::kFull) {
      pad_r[0] = GetPaddingSizeFull(data_md.data.dims[2], pad_l[0], pad_r[0], kernel[0], strides[0]);
      pad_r[1] = GetPaddingSizeFull(data_md.data.dims[3], pad_l[1], pad_r[1], kernel[1], strides[1]);
    }

    CHECK_GT(kernel[0], 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel[1], 0) << "Filter dimensions cannot be zero.";
  } else {
    CHECK_GE(param.pad.ndim(), 3);
    CHECK_GE(param.stride.ndim(), 3);
    kernel[0] = param.kernel[0];
    kernel[1] = param.kernel[1];
    kernel[2] = param.kernel[2];
    pad_l[0] = param.pad[0];
    pad_l[1] = param.pad[1];
    pad_l[2] = param.pad[2];
    pad_r[0] = param.pad[0];
    pad_r[1] = param.pad[1];
    pad_r[2] = param.pad[2];
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
    strides[2] = param.stride[2];

    if (param.global_pool) {
      kernel[0] = data_md.data.dims[2];
      kernel[1] = data_md.data.dims[3];
      kernel[2] = data_md.data.dims[4];
      strides[0] = 1;
      strides[1] = 1;
      strides[2] = 1;
      pad_l[0] = 0;
      pad_l[1] = 0;
      pad_l[2] = 0;
      pad_r[0] = 0;
      pad_r[1] = 0;
      pad_r[2] = 0;
    } else if (param.pooling_convention == pool_enum::kFull) {
      pad_r[0] = GetPaddingSizeFull(data_md.data.dims[2], pad_l[0], pad_r[0], kernel[0], strides[0]);
      pad_r[1] = GetPaddingSizeFull(data_md.data.dims[3], pad_l[1], pad_r[1], kernel[1], strides[1]);
      pad_r[2] = GetPaddingSizeFull(data_md.data.dims[4], pad_l[2], pad_r[2], kernel[2], strides[2]);
    }
  }
  if (pad_l[0] != 0 || pad_l[1] != 0) {
    CHECK(param.pool_type == pool_enum::kAvgPooling ||
          param.pool_type == pool_enum::kMaxPooling)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_l[0], kernel[0]);
    CHECK_LT(pad_l[1], kernel[1]);
  }
}

mkldnn::pooling_forward::primitive_desc GetPoolingFwdPdesc(
    const PoolingParam &param, const bool is_train, const mkldnn::memory::desc &data_md,
    const mkldnn::memory::desc &out_md) {
  CHECK(param.kernel.ndim() == 2 || param.kernel.ndim() == 3)
        << "Not Implemented";

  const int kernel_ndims = param.kernel.ndim();
  mkldnn::memory::dims kernel(kernel_ndims);
  mkldnn::memory::dims strides(kernel_ndims);
  // pad_l can be in 2d with (pad_t, pad_l) or 3d with (pad_b, pad_t, pad_l)
  mkldnn::memory::dims pad_l(kernel_ndims);
  // pad_r can be in 2d with (pad_b, pad_r) or 3d with (pad_a, pad_b, pad_r)
  mkldnn::memory::dims pad_r(kernel_ndims);

  // int kernel_h_, kernel_w_;
  InitPoolingPrimitiveParams(param, data_md, kernel, strides, pad_l, pad_r);

  const mkldnn::algorithm alg = GetMKLDNNPoolAlgo(param);
  mkldnn::prop_kind kind = mkldnn::prop_kind::forward_scoring;
  if (is_train && alg != mkldnn::algorithm::pooling_avg) {
    kind = mkldnn::prop_kind::forward_training;
  }

  const mkldnn::pooling_forward::desc poolingFwd_desc(kind, alg, data_md, out_md, strides,
                                                      kernel, pad_l, pad_r);
  return mkldnn::pooling_forward::primitive_desc(poolingFwd_desc, CpuEngine::Get()->get_engine());
}

MKLDNNPoolingFwd &GetPoolingFwd(const PoolingParam &param,
                                const bool is_train,
                                const NDArray &data,
                                const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNPoolingSignature,
                                         MKLDNNPoolingFwd,
                                         OpHash> pooling_fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNPoolingSignature,
                                            MKLDNNPoolingFwd,
                                            OpHash> pooling_fwds;
#endif

  bool with_workspace = is_train && MKLDNNRequireWorkspace(param);
  MKLDNNPoolingSignature key(param);
  key.AddSign(is_train);
  key.AddSign(with_workspace);
  key.AddSign(data);
  key.AddSign(output);

  auto it = pooling_fwds.find(key);
  if (it == pooling_fwds.end()) {
    CHECK(param.kernel.ndim() == 2 || param.kernel.ndim() == 3)
          << "Not Implemented";
    auto data_md = data.GetMKLDNNData()->get_desc();

    const int kernel_ndims = param.kernel.ndim();
    mkldnn::memory::dims kernel(kernel_ndims);
    mkldnn::memory::dims strides(kernel_ndims);
    // pad_l can be in 2d with (pad_t, pad_l) or 3d with (pad_b, pad_t, pad_l)
    mkldnn::memory::dims pad_l(kernel_ndims);
    // pad_r can be in 2d with (pad_b, pad_r) or 3d with (pad_a, pad_b, pad_r)
    mkldnn::memory::dims pad_r(kernel_ndims);
    InitPoolingPrimitiveParams(param, data_md, kernel, strides, pad_l, pad_r);

    const mkldnn::algorithm alg = GetMKLDNNPoolAlgo(param);
    MKLDNNPoolingFwd fwd(data, output, kernel, strides,
                         pad_l, pad_r, alg, with_workspace, is_train);
    it = AddToCache(&pooling_fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNPoolingCompute(const OpContext &ctx, const PoolingParam &param,
                          const NDArray &in_data, const OpReqType req,
                          const NDArray &out_data, const NDArray *workspace) {
  auto &fwd = GetPoolingFwd(param, ctx.is_train, in_data, out_data);
  fwd.Execute(in_data, req, out_data, workspace);
}

MKLDNNPoolingBwd::MKLDNNPoolingBwd(
    const mkldnn::pooling_backward::primitive_desc &pdesc, bool with_ws)
    : with_workspace(with_ws), pd(pdesc) {
      bwd = std::make_shared<mkldnn::pooling_backward>(pd);
    }

const mkldnn::pooling_backward &MKLDNNPoolingBwd::GetBwd() {
  return *this->bwd;
}

MKLDNNPoolingBwd &GetPoolingBwd(const PoolingParam &param,
                                const NDArray &in_data,
                                const NDArray &in_grad,
                                const NDArray &out_grad) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local
      std::unordered_map<MKLDNNPoolingSignature,
                         MKLDNNPoolingBwd, OpHash> pooling_bwds;
#else
  static MX_THREAD_LOCAL
      std::unordered_map<MKLDNNPoolingSignature,
                         MKLDNNPoolingBwd, OpHash> pooling_bwds;
#endif

  bool with_workspace = MKLDNNRequireWorkspace(param);
  MKLDNNPoolingSignature key(param);
  key.AddSign(in_data);
  key.AddSign(in_grad);
  key.AddSign(out_grad);

  auto it = pooling_bwds.find(key);
  if (it == pooling_bwds.end()) {
    NDArray diff_dst_buff = out_grad;
    if (in_data.IsMKLDNNData() == false && diff_dst_buff.IsMKLDNNData() == true) {
      diff_dst_buff = out_grad.Reorder2Default();
    }
    auto diff_dst_mem = diff_dst_buff.GetMKLDNNData();
    auto input_mem = in_data.GetMKLDNNData();
    const mkldnn::memory::desc data_md = input_mem->get_desc();
    
    // mkldnn::memory::dims out_dims;
    // if (out_grad.shape().ndim() == 4) {
    //   out_dims.push_back({data_md.data.dims[0], data_md.data.dims[1],
    //                       static_cast<int>(out_grad.shape()[2]),
    //                       static_cast<int>(out_grad.shape()[3])});
    // } else {
    //   out_dims.push_back({data_md.data.dims[0], data_md.data.dims[1],
    //                       static_cast<int>(out_grad.shape()[2]),
    //                       static_cast<int>(out_grad.shape()[3]),
    //                       static_cast<int>(out_grad.shape()[4])});
    // }
    // const mkldnn::memory::dims dims = {data_md.data.dims[0], data_md.data.dims[1],
    //                            static_cast<int>(out_grad.shape()[2]),
    //                            static_cast<int>(out_grad.shape()[3])};
    // const mkldnn::memory::desc out_md(
    //     {out_dims}, static_cast<mkldnn::memory::data_type>(data_md.data.data_type),
    //     mkldnn::memory::format_tag::any);
    
    const mkldnn::memory::desc out_md = GetMemDesc(out_grad);
    auto fwd_pd = GetPoolingFwdPdesc(param, true, data_md, out_md);
    const mkldnn::memory::desc diff_md = diff_dst_mem->get_desc();

    // const mkldnn::memory::dims dims1 = {diff_md.data.dims[0], diff_md.data.dims[1],
    //                             static_cast<int>(in_grad.shape()[2]),
    //                             static_cast<int>(in_grad.shape()[3])};
    // mkldnn::memory::dims diff_dims;
    // if (in_grad.shape().ndim() == 4) {
    //   diff_dims.push_back({diff_md.data.dims[0], diff_md.data.dims[1],
    //                        static_cast<int>(in_grad.shape()[2]),
    //                        static_cast<int>(in_grad.shape()[3])});
    // } else {
    //   diff_dims.push_back({diff_md.data.dims[0], diff_md.data.dims[1],
    //                        static_cast<int>(in_grad.shape()[2]),
    //                        static_cast<int>(in_grad.shape()[3]),
    //                        static_cast<int>(in_grad.shape()[4])});
    // }
    // const mkldnn::memory::desc diff_in_md(
    //     {diff_dims}, static_cast<mkldnn::memory::data_type>(diff_md.data.data_type),
    //     mkldnn::memory::format_tag::any);
    const mkldnn::memory::desc diff_in_md = GetMemDesc(in_grad);
    const mkldnn::engine cpu_engine = CpuEngine::Get()->get_engine();
    const mkldnn::algorithm alg = GetMKLDNNPoolAlgo(param);

    const int kernel_ndims = param.kernel.ndim();
    mkldnn::memory::dims kernel(kernel_ndims);
    mkldnn::memory::dims strides(kernel_ndims);
    // pad_l can be in 2d with (pad_t, pad_l) or 3d with (pad_b, pad_t, pad_l)
    mkldnn::memory::dims pad_l(kernel_ndims);
    // pad_r can be in 2d with (pad_b, pad_r) or 3d with (pad_a, pad_b, pad_r)
    mkldnn::memory::dims pad_r(kernel_ndims);

    InitPoolingPrimitiveParams(param, data_md, kernel, strides, pad_l, pad_r);

    const mkldnn::pooling_backward::desc desc(
                alg, diff_in_md, diff_md, strides, kernel, pad_l, pad_r);
    const auto pdesc = mkldnn::pooling_backward::primitive_desc(desc, cpu_engine, fwd_pd);
    MKLDNNPoolingBwd bwd(pdesc, with_workspace);
    it = AddToCache(&pooling_bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNPoolingGradCompute(const OpContext &ctx, const PoolingParam &param,
                              const NDArray &out_grad, const NDArray &in_data,
                              const NDArray *workspace, const OpReqType req,
                              const NDArray &in_grad) {
  if (req == kNullOp) {
    return;
  }
  TmpMemMgr::Get()->Init(ctx.requested[0]);

  auto &bwd = GetPoolingBwd(param, in_data, in_grad, out_grad);
  auto diff_src_mem =
      CreateMKLDNNMem(in_grad, bwd.pd.diff_src_desc(), req);

  mkldnn_args_map_t args = {
    {MKLDNN_ARG_DIFF_DST, *(out_grad.GetMKLDNNData())},
    {MKLDNN_ARG_DIFF_SRC, *diff_src_mem.second },
  };
  if (MKLDNNRequireWorkspace(param) && workspace != nullptr) {
    args[MKLDNN_ARG_WORKSPACE] = *(workspace->GetMKLDNNData());
  }

  MKLDNNStream::Get()->RegisterPrimArgs(bwd.GetBwd(), args);
  CommitOutput(in_grad, diff_src_mem);
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
