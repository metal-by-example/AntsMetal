
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
        vertexDescriptor.attributes[1].format = .float2 // texture coordinates
        vertexDescriptor.attributes[1].offset = MemoryLayout<Float>.size * 2
        vertexDescriptor.attributes[1].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<Float>.size * 4
        
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
        
        if let kernelFunction = library.makeFunction(name: "antsKernel") {
            computePipelineState = try? device.makeComputePipelineState(function: kernelFunction)
        }

        createTextures()

        (view as? MTKView)?.preferredFramesPerSecond = 30
    }

    // MARK: - Generate Textures

    func createTextures() {
        textures.removeAll()

        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.width = Int(view.frame.width)
        textureDescriptor.height = Int(view.frame.height)
        textureDescriptor.pixelFormat = .bgra8Unorm
        textureDescriptor.resourceOptions = .storageModePrivate
        textureDescriptor.usage = [.shaderWrite, .shaderRead, .renderTarget]

        textures.append(device.makeTexture(descriptor: textureDescriptor)!)
        textures.append(device.makeTexture(descriptor: textureDescriptor)!)
        
        // Run an empty render pass to clear the contents of the texture we'll be blitting from first
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = textures[inactiveTextureIndex]
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1)
        let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        renderCommandEncoder.endEncoding()
        commandBuffer.commit()
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

        // Copy the contents of the previously-current texture to the texture we're about to read from
        let blitCommandEncoder = commandBuffer.makeBlitCommandEncoder()!
        blitCommandEncoder.copy(from: textures[inactiveTextureIndex],
                                sourceSlice: 0,
                                sourceLevel: 0,
                                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                                sourceSize: MTLSize(width: textures[inactiveTextureIndex].width,
                                                    height: textures[inactiveTextureIndex].height,
                                                    depth: 1),
                                to: textures[currentTextureIndex],
                                destinationSlice: 0,
                                destinationLevel: 0,
                                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blitCommandEncoder.endEncoding()

        // MARK: - Compute Work Encoding

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(computePipelineState)
        // pass current time as seed for rng
        computeEncoder.setBytes(&time, length: MemoryLayout<Float>.stride, index: 0)
        // Set alternating in/out textures
        computeEncoder.setTexture(textures[inactiveTextureIndex], index: 0)
        computeEncoder.setTexture(textures[currentTextureIndex], index: 1)

        let threadsPerThreadGroup = MTLSizeMake(16, 16, 1)

        // The algorithm that generates the pattern only requires that we run on a quarter of the pixels
        // See in the antsKernel function in Shaders.metal how the pixel coordinates are multiplied by 2
        // So as far as I understood it the below code makes it so that the kernel function only executes
        // on the top left quadrant of the texture (but modifies the entire texture)
        let threadGroups = MTLSizeMake(
            textures[currentTextureIndex].width / 2,
            textures[currentTextureIndex].height / 2,
            1)
        computeEncoder.dispatchThreads(threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
        computeEncoder.endEncoding()



        
        let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!

        renderCommandEncoder.setRenderPipelineState(renderPipelineState)
        
        let rect = Rect2D(origin: Point2D(x: 0.0, y: 0.0),
                          size: Size2D(width: Float(view.drawableSize.width),
                                       height: Float(view.drawableSize.height)))
        
        // Corners of a screen-space quad (with +Y going down), suitable for
        // drawing as a tri strip with CCW winding. Total data size = 64 bytes
        var vertexData: [Float] = [
            rect.minX, rect.minY, 0, 0, // top left
            rect.minX, rect.maxY, 0, 1, // bottom left
            rect.maxX, rect.minY, 1, 0, // top right
            rect.maxX, rect.maxY, 1, 1, // bottom right
        ]
        
        var projectionMatrix = float4x4(orthoProjectionWidth: Float(view.drawableSize.width),
                                        height: Float(view.drawableSize.height),
                                        zNear: 0,
                                        zFar: 1.0)

        // MARK: - Draw Call Encoding

        renderCommandEncoder.setVertexBytes(&vertexData, length: vertexData.count * MemoryLayout<Float>.size, index: 0)
        renderCommandEncoder.setVertexBytes(&projectionMatrix, length: MemoryLayout<float4x4>.size, index: 1)
        renderCommandEncoder.setFragmentTexture(textures[currentTextureIndex], index: 0)
        
        renderCommandEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderCommandEncoder.endEncoding()
        
        commandBuffer.present(view.currentDrawable!)
        commandBuffer.commit()

        // MARK: - Updating Current Texture
        // Switch out the current texture
        swap(&currentTextureIndex, &inactiveTextureIndex)

        time += 1 / Float(view.preferredFramesPerSecond)
    }
}

