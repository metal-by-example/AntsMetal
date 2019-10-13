
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 position [[attribute(0)]];
    float2 texCoords [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoords;
};

// Produces 3 float values between [0, 1] based on a 3-uint seed
// Inspired by https://www.shadertoy.com/view/XlXcW4
float3 random(uint3 v) {
    const uint k = 1103515245U; // iq
    v = ((v >> 8U) ^ v.yzx) * k;
    v = ((v >> 8U) ^ v.yzx) * k;
    v = ((v >> 8U) ^ v.yzx) * k;
    return float3(v) * (1.0f / float(0xffffffffU));
}

// Generate a random float in the range [0.0f, 1.0f] using x, y, and z (based on the xor128 algorithm)
float rand(int x, int y, int z)
{
    int seed = x + y * 57 + z * 241;
    seed= (seed<< 13) ^ seed;
    return (( 1.0 - ( (seed * (seed * seed * 15731 + 789221) + 1376312589) & 2147483647) / 1073741824.0f) + 1.0f) / 2.0f;
}

kernel void
antsKernel(texture2d<half, access::read>  inTexture  [[texture(0)]],
           texture2d<half, access::write> outTexture [[texture(1)]],
           uint2                          gid        [[thread_position_in_grid]],
           constant float&                time       [[buffer(0)]])
{
    // This formula is based on a very very very old Processing thing I randomly wrote.
    // The original formula and a question on how to implement this (posted over 8 years ago)
    // can be found here on StackOverflow:
    // https://stackoverflow.com/questions/4765596/loop-through-all-pixels-and-get-set-individual-pixel-color-in-opengl
    uint width = inTexture.get_width();
    uint height = inTexture.get_height();
    float3 r = random(uint3(gid.x, gid.y, time * 191));
    uint newX = 2 * gid.x + round(r.x);
    uint newY = 2 * gid.y + round(r.y);

    if (newX < width && newY < height) {
        uint2 destinationPixel = uint2(newX, newY);
        half4 sourcePixelColor = inTexture.read(gid);
        half4 outColor;
        if (sourcePixelColor.r == 1) {
            outColor = half4(0, 0, 0, 1);
        } else {
            outColor = half4(1, 1, 1, 1);
        }

        outTexture.write(outColor, destinationPixel);
    } else {
        outTexture.write(half4(1, 0, 1, 1), gid);
    }
}


vertex VertexOut tiled_textured_vertex(VertexIn in [[stage_in]],
                                       constant float4x4& projectionMatrix [[buffer(1)]])
{
    VertexOut out;
    out.position = projectionMatrix * float4(in.position.xy, 0.0f, 1.0f);
    out.texCoords = in.texCoords;
    return out;
}

fragment float4 tiled_textured_fragment(VertexOut in [[stage_in]],
                                        texture2d<float, access::sample> tex2d [[texture(0)]])
{
    constexpr sampler textureSampler (mag_filter::nearest);
    return tex2d.sample(textureSampler, in.texCoords);
}
