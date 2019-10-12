
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 position [[attribute(0)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoords;
};

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
    uint newX = 2 * gid.x + rand(time, gid.y, time);
    uint newY = 2 * gid.y + rand(time, gid.x, time);

    if (newX < width && newY < height) {
        uint2 destinationPixel = uint2(newX, newY);
        half4 sourcePixelColor = inTexture.read(gid);
        half4 outColor;
        if (sourcePixelColor.r == 1) {
            outColor = half4(0, 0, 0, 0);
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
    out.position = float4(in.position.xy, 0.0f, 1.0f);
    out.texCoords = in.position.xy;
    return out;
}

fragment float4 tiled_textured_fragment(VertexOut in [[stage_in]],
                                        texture2d<float, access::sample> tex2d [[texture(0)]])
{
    constexpr sampler textureSampler (mag_filter::linear, min_filter::linear);
    return tex2d.sample(textureSampler, in.texCoords);
}
