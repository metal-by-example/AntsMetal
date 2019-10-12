
import Cocoa
import Metal
import MetalKit

struct Point2D {
    var x: Float = 0.0
    var y: Float = 0.0
}

struct Size2D {
    var width: Float = 0.0
    var height: Float = 0.0
}

struct Rect2D {
    var origin = Point2D()
    var size = Size2D()
}

extension simd_float4x4 {
    // Adapted from https://docs.microsoft.com/en-us/windows/win32/direct3d9/d3dxmatrixorthooffcenterrh
    init(orthoProjectionWidth w: Float, height h: Float, zNear zn: Float, zFar zf: Float) {
        self.init(
            SIMD4<Float>( 2.0 / w,      0.0,            0.0, 0.0),
            SIMD4<Float>(     0.0, 2.0 / -h,            0.0, 0.0),
            SIMD4<Float>(     0.0,      0.0,  1 / (zn - zf), 0.0),
            SIMD4<Float>(    -1.0,      1.0, zn / (zn - zf), 1.0)
        )
    }
}

extension Rect2D {
    var minX: Float { return origin.x }
    var maxX: Float { return origin.x + size.width }
    var minY: Float { return origin.y }
    var maxY: Float { return origin.y + size.height }
    var width: Float { return size.width }
    var height: Float { return size.height }
}

class ViewController: NSViewController, MTKViewDelegate {
    
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var renderPipelineState: MTLRenderPipelineState!
    var computePipelineState: MTLComputePipelineState!
    var textures: [MTLTexture] = []
    var currentTextureIndex = 0
    var inactiveTextureIndex = 1
    var time: Float = 0.0
    var library: MTLLibrary!

    override func viewDidLoad() {
        super.viewDidLoad()
        
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device.makeCommandQueue()!
        
        let metalView = view as! MTKView
        metalView.device = device
        metalView.colorPixelFormat = .bgra8Unorm_srgb
        metalView.clearColor = MTLClearColorMake(1.0, 1.0, 1.0, 1.0)
        metalView.delegate = self
        
        library = device.makeDefaultLibrary()!
        let vertexFunction = library.makeFunction(name: "tiled_textured_vertex")
        let fragmentFunction = library.makeFunction(name: "tiled_textured_fragment")
        
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2 // screen-space position
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<Float>.size * 2
        
        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineDescriptor.vertexFunction = vertexFunction
        renderPipelineDescriptor.fragmentFunction = fragmentFunction
        renderPipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        renderPipelineDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        
        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        } catch {
            print("Filed to create render pipeline state because \(error)")
        }

        createTextures()

        (view as? MTKView)?.preferredFramesPerSecond = 1
    }

    // MARK: - Generate Textures

    func createTextures() {
        textures.removeAll()

        let textureDescriptor = MTLTextureDescriptor()
        // Got an error saying textures had to be divisible by 256
        textureDescriptor.width = Int(ceil(view.frame.width / 256)) * 256
        textureDescriptor.height = Int(ceil(view.frame.height / 256)) * 256
        textureDescriptor.pixelFormat = .bgra8Unorm
        // Got an error about resource options had to match the buffer resource options. Don't know if this is correct.
        textureDescriptor.resourceOptions = .storageModePrivate
        textureDescriptor.usage = [.shaderWrite, .shaderRead]

        let floatSize = MemoryLayout<Float>.stride
        let count = textureDescriptor.width * 2 * textureDescriptor.height * 2
        let textureBuffer = device.makeBuffer(length: count * floatSize, options: .storageModePrivate)!

        textures.append(textureBuffer.makeTexture(
            descriptor: textureDescriptor,
            offset: 0,
            bytesPerRow: textureDescriptor.width * 4 * floatSize)!)
        textures.append(textureBuffer.makeTexture(
            descriptor: textureDescriptor,
            offset: 0,
            bytesPerRow: textureDescriptor.width * 4)!)
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        createTextures()
    }
    
    func draw(in view: MTKView) {
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        guard let renderPassDescriptor = view.currentRenderPassDescriptor else {
            print("MTKView didn't have a drawable available; dropping frame")
            return
        }


        // MARK: - Kernel function

        // I'm not even sure this is the right place to do this.
        guard
            let kernelFunction = library.makeFunction(name: "antsKernel"),
            let computePipelineState = try? device.makeComputePipelineState(function: kernelFunction),
            let computeEncoder = commandBuffer.makeComputeCommandEncoder()
            else { return }
        computeEncoder.setComputePipelineState(computePipelineState)
        // pass current time as seed for rng
        computeEncoder.setBytes(&time, length: MemoryLayout<Float>.stride, index: 0)
        // Set alternating in/out textures
        computeEncoder.setTexture(textures[currentTextureIndex], index: 0)
        computeEncoder.setTexture(textures[inactiveTextureIndex], index: 1)

        let threadsPerThreadGroup = MTLSizeMake(16, 16, 1)

        // The algorithm that generates the pattern only requires that we run on a quarter of the pixels
        // See in the antsKernel function in Shaders.metal how the pixel coordinates are multiplied by 2
        // So as far as I understood it the below code makes it so that the kernel function only executes
        // on the top left quadrant of the texture (but modifies the entire texture)
        let threadGroups = MTLSizeMake(
            textures[currentTextureIndex].width / 2 / threadsPerThreadGroup.width,
            textures[currentTextureIndex].height / 2 / threadsPerThreadGroup.height,
            1)
        computeEncoder.dispatchThreads(threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
        computeEncoder.endEncoding()



        
        let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!

        renderCommandEncoder.setRenderPipelineState(renderPipelineState)
        
        let rect = Rect2D(origin: Point2D(x: 0.0, y: 0.0),
                          size: Size2D(width: Float(view.drawableSize.width),
                                       height: Float(view.drawableSize.height)))
        
        // Corners of a screen-space quad (with +Y going down), suitable for
        // drawing as a tri strip with CCW winding. Total data size = 32 bytes
        var vertexData: [Float] = [
            rect.minX, rect.minY, // top left
            rect.minX, rect.maxY, // bottom left
            rect.maxX, rect.minY, // top right
            rect.maxX, rect.maxY  // bottom right
        ]
        
        var projectionMatrix = float4x4(orthoProjectionWidth: Float(view.drawableSize.width),
                                        height: Float(view.drawableSize.height),
                                        zNear: 0,
                                        zFar: 1.0)

        // MARK: - Texture Scaling
        // I removed the code for texture scaling since this texture should cover the entire screen it doesn't need to scale.
        // Maybe?


        renderCommandEncoder.setVertexBytes(&vertexData, length: vertexData.count * MemoryLayout<Float>.size, index: 0)
        renderCommandEncoder.setVertexBytes(&projectionMatrix, length: MemoryLayout<float4x4>.size, index: 1)
        renderCommandEncoder.setFragmentTexture(textures[currentTextureIndex], index: 0)
        
        renderCommandEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderCommandEncoder.endEncoding()
        
        commandBuffer.present(view.currentDrawable!)
        commandBuffer.commit()

        // MARK: - Updating Current Texture
        // Switch out the current texture
        currentTextureIndex = currentTextureIndex == 1 ? 0 : 1
        inactiveTextureIndex = currentTextureIndex == 1 ? 0 : 1

        time += 1 / Float(view.preferredFramesPerSecond)
    }
}

