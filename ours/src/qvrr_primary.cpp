#include "ours/qvrr_primary.h"
#include "ours/fp32_inner_product.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <limits>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace ours {
namespace {
template<class T> T Read(const uint8_t* p) { T x; std::memcpy(&x,p,sizeof x); return x; }
float Half(const uint8_t* p) {
  uint16_t h=Read<uint16_t>(p); unsigned e=(h>>10)&31, m=h&1023;
  if (e==31) throw std::runtime_error("Nonfinite LVQ bound");
  float x=e ? std::ldexp(1.0f+float(m)/1024,int(e)-15) : std::ldexp(float(m),-24);
  return (h&32768) ? -x : x;
}
#if defined(__AVX512F__)
// Gather only complete 4-byte words WITHIN each record. The final byte may
// require a shifted word ending at record_size-1, never an out-of-file load.
struct RecordBatch {
  const uint8_t* data;
  __mmask16 mask;
  bool wide;
  __m512i offsets32=_mm512_setzero_si512(), offsets_lo=_mm512_setzero_si512(), offsets_hi=_mm512_setzero_si512();
  RecordBatch(const uint8_t* p, const uint32_t* ids, size_t count,
              uint32_t stride, uint64_t payload):data(p),
      mask(count==16 ? 0xffff : static_cast<__mmask16>((1U<<count)-1)),
      wide(payload>uint64_t(std::numeric_limits<int32_t>::max())) {
    const __m512i v=_mm512_maskz_loadu_epi32(mask,ids);
    if (!wide) offsets32=_mm512_mullo_epi32(v,_mm512_set1_epi32(stride));
    else {
      offsets_lo=_mm512_mullo_epi64(_mm512_cvtepu32_epi64(_mm512_castsi512_si256(v)),_mm512_set1_epi64(stride));
      offsets_hi=_mm512_mullo_epi64(_mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(v,1)),_mm512_set1_epi64(stride));
    }
  }
  __m512i Word(uint32_t at) const {
    if (!wide) return _mm512_mask_i32gather_epi32(_mm512_setzero_si512(),mask,
        _mm512_add_epi32(offsets32,_mm512_set1_epi32(at)),data,1);
    const auto delta=_mm512_set1_epi64(at);
    auto lo=_mm512_mask_i64gather_epi32(_mm256_setzero_si256(),static_cast<__mmask8>(mask),
        _mm512_add_epi64(offsets_lo,delta),data,1);
    auto hi=_mm512_mask_i64gather_epi32(_mm256_setzero_si256(),static_cast<__mmask8>(mask>>8),
        _mm512_add_epi64(offsets_hi,delta),data,1);
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo),hi,1);
  }
  __m512i Byte(uint32_t at, uint32_t stride) const {
    const uint32_t word=std::min(at&~3U,stride-4);
    return _mm512_and_si512(_mm512_srli_epi32(Word(word),(at-word)*8),_mm512_set1_epi32(255));
  }
};
#endif
}
struct QvrrPrimaryIndex::Impl {
  struct Segment { uint32_t start,width,bits,bit_offset,factor_offset; float divisor; };
  uint64_t n=0, model_bytes=0, size=0;
  uint32_t d=0, sd=0, record=0, groups=0, width=0, packed=0;
  std::string magic;
  std::vector<float> projection, books, mean;
  std::vector<Segment> segments;
  int fd=-1; const uint8_t* mapped=nullptr; const uint8_t* records=nullptr;
  ~Impl() { if(mapped) munmap(const_cast<uint8_t*>(mapped),size); if(fd>=0) close(fd); }
};
QvrrPrimaryIndex::QvrrPrimaryIndex(const std::string& mp,const std::string& cp):impl_(new Impl) {
  RequireFp32Avx512();
  auto& a=*impl_; std::ifstream f(mp,std::ios::binary);
  if(!f) throw std::runtime_error("Cannot open QVRR model");
  std::vector<uint8_t> m((std::istreambuf_iterator<char>(f)),{});
  if(m.size()<44) throw std::runtime_error("Truncated QVRR model");
  a.model_bytes=m.size(); a.magic.assign(reinterpret_cast<char*>(m.data()),8);
  const auto u=[&](size_t off){ if(off+4>m.size()) throw std::runtime_error("Truncated model field"); return Read<uint32_t>(m.data()+off); };
  if(u(8)!=1) throw std::runtime_error("Unsupported QVRR version");
  a.n=Read<uint64_t>(m.data()+12); a.d=u(20);
  if(!a.n || !a.d || a.d>65536) throw std::runtime_error("Invalid QVRR dimensions");
  size_t offset=0, expected=0; bool column_major=false;
  if(a.magic=="OPQPQ001") {
    a.sd=u(24); a.groups=u(28); a.width=u(36); a.record=a.groups;
    if(u(32)!=256 || uint64_t(a.groups)*a.width!=a.sd) throw std::runtime_error("Unsupported PQ layout");
    offset=48; expected=offset+4ULL*a.d*a.sd+4ULL*a.groups*256*a.width;
  } else if(a.magic=="ERBQRG01") {
    a.sd=a.d; unsigned bits=u(24); a.packed=u(28); a.record=a.packed+4;
    if(!bits || bits>8 || a.packed!=(uint64_t(a.d)*bits+7)/8) throw std::runtime_error("Invalid RaBitQ packing");
    a.segments.push_back({0,a.d,bits,0,a.packed,1}); offset=44; expected=offset+4ULL*a.d*a.sd;
  } else if(a.magic=="SAQRG001") {
    a.sd=u(24); unsigned ns=u(32); offset=48+12ULL*ns; uint32_t bit=0;
    if(ns>1024 || offset>m.size()) throw std::runtime_error("Invalid SAQ segments");
    for(unsigned g=0;g<ns;++g) {
      unsigned s=u(48+12*g), w=u(52+12*g), b=u(56+12*g);
      if(b>8 || uint64_t(s)+w>a.sd) throw std::runtime_error("Invalid SAQ segment");
      a.segments.push_back({s,w,b,bit,g,b ? std::ldexp(1.0f,int(b)-1):1}); bit+=w*b;
    }
    a.packed=(bit+7)/8; a.record=a.packed+4*ns;
    for(auto& seg:a.segments) seg.factor_offset=a.packed+4*seg.factor_offset;
    expected=offset+4ULL*a.d*a.sd;
  } else if(a.magic=="LVOOD001") {
    a.sd=u(24); a.record=u(28); offset=56; column_major=true;
    if(a.record!=a.sd+4) throw std::runtime_error("Invalid LeanVec record");
    expected=offset+4ULL*(a.d*a.sd+a.sd);
  } else throw std::runtime_error("Unsupported QVRR model magic");
  if(!a.sd || a.sd>65536 || expected!=m.size()) throw std::runtime_error("QVRR model size mismatch");
  // Store projection columns contiguous for the native vectorized dot kernel.
  a.projection.resize(uint64_t(a.d)*a.sd);
  for(unsigned j=0;j<a.sd;++j) for(unsigned k=0;k<a.d;++k)
    a.projection[uint64_t(j)*a.d+k]=Read<float>(m.data()+offset+4ULL*(column_major?uint64_t(j)*a.d+k:uint64_t(k)*a.sd+j));
  if(a.groups) {
    a.books.resize(uint64_t(a.groups)*256*a.width);
    std::memcpy(a.books.data(),m.data()+offset+4ULL*a.d*a.sd,4*a.books.size());
  } else if(column_major) {
    a.mean.resize(a.sd); std::memcpy(a.mean.data(),m.data()+offset+4ULL*a.d*a.sd,4*a.sd);
  }
  for(float v:a.projection) if(!std::isfinite(v)) throw std::runtime_error("Nonfinite QVRR transform");
  a.fd=open(cp.c_str(),O_RDONLY); struct stat st{};
  if(a.fd<0 || fstat(a.fd,&st)!=0) throw std::runtime_error("Cannot open QVRR codes");
  a.size=st.st_size;
  if(a.size!=4096+a.n*a.record) throw std::runtime_error("QVRR code length mismatch");
  void* mapping=mmap(nullptr,a.size,PROT_READ,MAP_PRIVATE,a.fd,0);
  if(mapping==MAP_FAILED) throw std::runtime_error("Cannot map QVRR codes");
  a.mapped=static_cast<const uint8_t*>(mapping); a.records=a.mapped+4096;
  std::string magic(reinterpret_cast<const char*>(a.mapped),8);
  auto cu=[&](size_t off){return Read<uint32_t>(a.mapped+off);};
  if(Read<uint64_t>(a.mapped+12)!=a.n) throw std::runtime_error("QVRR node count mismatch");
  if(a.magic=="SAQRG001") {
    if(magic!=std::string("SAQC002\0",8) || cu(8)!=2 || cu(20)!=a.sd || cu(24)!=a.segments.size() || cu(28)!=a.packed || cu(32)!=a.record || cu(36)!=4096) throw std::runtime_error("SAQ code header mismatch");
  } else {
    std::string want=a.groups?std::string("OPQC001\0",8):a.magic=="LVOOD001"?std::string("LVC8001\0",8):"ERBQC001";
    unsigned count=a.groups?a.groups:a.magic=="LVOOD001"?a.sd:a.packed;
    bool lean=a.magic=="LVOOD001";
    if(magic!=want || cu(8)!=1 || cu(20)!=count || cu(24)!=a.record || cu(lean?32:28)!=4096 || (lean && cu(28)!=8)) throw std::runtime_error("QVRR code header mismatch");
  }
}
QvrrPrimaryIndex::~QvrrPrimaryIndex()=default;
uint64_t QvrrPrimaryIndex::nodes() const{return impl_->n;}
uint32_t QvrrPrimaryIndex::original_dim() const{return impl_->d;}
uint32_t QvrrPrimaryIndex::scorer_dim() const{return impl_->sd;}
uint64_t QvrrPrimaryIndex::storage_bytes() const{return impl_->size+impl_->model_bytes;}
const char* QvrrPrimaryIndex::scoring_kernel() const {
#if defined(__AVX512F__)
  if(impl_->groups) return "avx512-qvrr-pq8-candidate16";
  if(!impl_->mean.empty()) return "avx512-qvrr-leanvec-lvq8-candidate16";
  if(impl_->magic=="SAQRG001") return "avx512-qvrr-saq-packed-candidate16";
  return "avx512-qvrr-rabitq-packed-candidate16";
#else
  return "qvrr-portable-scalar";
#endif
}
QvrrPrimaryIndex::QueryContext QvrrPrimaryIndex::Prepare(const float* q,uint32_t d) const {
  const auto& a=*impl_; if(!q || d!=a.d) throw std::invalid_argument("QVRR raw query dimension mismatch");
  QueryContext c; c.transformed.resize(a.sd);
  for(unsigned j=0;j<a.sd;++j) c.transformed[j]=Fp32InnerProduct(q,a.projection.data()+uint64_t(j)*a.d,a.d);
  if(a.groups) {
    c.lut.resize(uint64_t(a.groups)*256);
    for(unsigned g=0;g<a.groups;++g) for(unsigned k=0;k<256;++k)
      c.lut[g*256+k]=Fp32InnerProduct(c.transformed.data()+g*a.width,a.books.data()+uint64_t(g*256+k)*a.width,a.width);
  } else if(!a.mean.empty()) {
    c.bias=Fp32InnerProduct(c.transformed.data(),a.mean.data(),a.sd);
    for(float v:c.transformed) c.transformed_sum+=v;
  }
  return c;
}
float QvrrPrimaryIndex::Score(const QueryContext& c,uint32_t id) const {
  const auto& a=*impl_; if(id>=a.n) throw std::out_of_range("QVRR node ID");
  const uint8_t* r=a.records+uint64_t(id)*a.record; float s=c.bias;
  if(a.groups) { for(unsigned g=0;g<a.groups;++g) s+=c.lut[g*256+r[g]]; }
  else if(!a.mean.empty()) {
    float lo=Half(r), hi=Half(r+2), dot=0, sum=0;
    for(unsigned j=0;j<a.sd;++j) { dot+=c.transformed[j]*r[4+j]; sum+=c.transformed[j]; }
    s+=lo*sum+(hi-lo)/255.0f*dot;
  } else {
    for(const auto& seg:a.segments) {
      if(!seg.bits) continue;
      float t=0; unsigned mask=(1U<<seg.bits)-1; float mid=float(mask)/2;
      for(unsigned j=0;j<seg.width;++j) {
        unsigned bit=seg.bit_offset+j*seg.bits, at=bit/8, shift=bit%8;
        unsigned z=r[at]; if(shift+seg.bits>8) z|=unsigned(r[at+1])<<8;
        t+=c.transformed[seg.start+j]*(float((z>>shift)&mask)-mid);
      }
      float factor=Read<float>(r+seg.factor_offset);
      if(!std::isfinite(factor)) throw std::runtime_error("Nonfinite quantized factor");
      s+=t*(factor/seg.divisor);
    }
  }
  return s;
}
void QvrrPrimaryIndex::ScoreBatch(const QueryContext& c,const uint32_t* ids,size_t n,float* out) const {
  const auto& a=*impl_;
  if(!n) return;
  if(!ids || !out) throw std::invalid_argument("Null QVRR batch");
  for(size_t i=0;i<n;++i) if(ids[i]>=a.n) throw std::out_of_range("QVRR node ID");
  size_t i=0;
#if defined(__AVX512F__)
  // Very small batches retain the scalar reference to avoid gather overhead.
  if(a.record>=4) for(; i+4<=n;) {
    const size_t count=std::min<size_t>(16,n-i);
    RecordBatch r(a.records,ids+i,count,a.record,a.n*a.record);
    __m512 score=_mm512_set1_ps(c.bias);
    if(a.groups) {
      for(unsigned g=0;g<a.groups;g+=4) {
        const unsigned at=std::min(g,a.record-4);
        const auto word=r.Word(at);
        for(unsigned j=g;j<std::min(g+4,a.groups);++j) {
          const auto code=_mm512_and_si512(_mm512_srli_epi32(word,(j-at)*8),_mm512_set1_epi32(255));
          score=_mm512_add_ps(score,_mm512_i32gather_ps(code,c.lut.data()+j*256,4));
        }
      }
    } else if(!a.mean.empty()) {
      const auto bounds=r.Word(0);
      if(_mm512_mask_cmpeq_epi32_mask(r.mask,_mm512_and_si512(bounds,_mm512_set1_epi32(0x7c00)),_mm512_set1_epi32(0x7c00)) ||
         _mm512_mask_cmpeq_epi32_mask(r.mask,_mm512_and_si512(bounds,_mm512_set1_epi32(0x7c000000)),_mm512_set1_epi32(0x7c000000)))
        throw std::runtime_error("Nonfinite LVQ bound");
      const auto lo=_mm512_cvtph_ps(_mm512_cvtepi32_epi16(bounds));
      const auto hi=_mm512_cvtph_ps(_mm512_cvtepi32_epi16(_mm512_srli_epi32(bounds,16)));
      auto dot=_mm512_setzero_ps();
      uint32_t previous=std::numeric_limits<uint32_t>::max();
      __m512i cached=_mm512_setzero_si512();
      for(unsigned j=0;j<a.sd;++j) {
        const unsigned at=std::min((4+j)&~3U,a.record-4);
        if(at!=previous) { cached=r.Word(at); previous=at; }
        const auto code=_mm512_and_si512(_mm512_srli_epi32(cached,(4+j-at)*8),_mm512_set1_epi32(255));
        dot=_mm512_fmadd_ps(_mm512_set1_ps(c.transformed[j]),_mm512_cvtepi32_ps(code),dot);
      }
      auto scale=_mm512_div_ps(_mm512_sub_ps(hi,lo),_mm512_set1_ps(255));
      auto value=_mm512_fmadd_ps(scale,dot,_mm512_mul_ps(lo,_mm512_set1_ps(c.transformed_sum)));
      score=_mm512_add_ps(score,value);
    } else {
      for(const auto& seg:a.segments) {
        if(!seg.bits) continue;
        const unsigned mask=(1U<<seg.bits)-1;
        const auto midpoint=_mm512_set1_ps(float(mask)/2);
        auto dot=_mm512_setzero_ps();
        uint32_t previous=std::numeric_limits<uint32_t>::max();
        __m512i cached=_mm512_setzero_si512();
        for(unsigned j=0;j<seg.width;++j) {
          const unsigned bit=seg.bit_offset+j*seg.bits, word=bit/32*4, shift=bit%32;
          if(word!=previous) { cached=r.Word(word); previous=word; }
          auto code=_mm512_srli_epi32(cached,shift);
          if(shift+seg.bits>32) code=_mm512_or_si512(code,_mm512_slli_epi32(r.Word(word+4),32-shift));
          code=_mm512_and_si512(code,_mm512_set1_epi32(mask));
          dot=_mm512_fmadd_ps(_mm512_set1_ps(c.transformed[seg.start+j]),
              _mm512_sub_ps(_mm512_cvtepi32_ps(code),midpoint),dot);
        }
        const auto factor_bits=r.Word(seg.factor_offset);
        if(_mm512_mask_cmpeq_epi32_mask(r.mask,_mm512_and_si512(factor_bits,_mm512_set1_epi32(0x7f800000)),_mm512_set1_epi32(0x7f800000)))
          throw std::runtime_error("Nonfinite quantized factor");
        auto factor=_mm512_div_ps(_mm512_castsi512_ps(factor_bits),_mm512_set1_ps(seg.divisor));
        score=_mm512_fmadd_ps(dot,factor,score);
      }
    }
    _mm512_mask_storeu_ps(out+i,r.mask,score); i+=count;
  }
#endif
  for(;i<n;++i) out[i]=Score(c,ids[i]);
}
}
