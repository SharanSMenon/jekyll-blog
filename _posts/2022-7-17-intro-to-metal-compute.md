---
title: "Getting started with GPU Compute on Metal"
date: 2022-7-17
toc: true
toc_label: "Contents"
toc_sticky: True
published: true
excerpt: "Create a Metal function that adds two large vectors."
categories:
  - Programming
  - GPU Programming
  - Apple
tags:
  - Objective-C
  - Metal
  - macOS
---

GPUs can be used for rendering complex games and running complicated calculations. In fact, GPUs are used to train ML models due to their extreme parallelism. This tutorial goes over the basics of graphics programming with a example on how to add 2 large vectors using Apple's Metal API.

> You will need a device that runs **macOS** for this.

## Setup

To get started, ensure that you have XCode and its developer tools installed and ready to go.

We will create a new project in XCode:

1. Create a new project in XCode
2. Go to the `macOS` tab in the create project menu
3. Select `Command Line Tool`
4. Change the language to `Objective-C`, and fill out any other information

> You can also do this in Swift, you will just have to slightly adapt the code from this tutorial.

That should be our setup, very simple. When you enter the project, you should see a `main.m`. This is our entry point and our code will execute from here.

We can start by creating a Metal Shader file to define our add shader. Create a file called `addShader.metal`.

```metal
#include <metal_stdlib>
using namespace metal;

kernel void addShader(device const float* inA,
                      device const float* inB,
                      device float* result,
                      uint index [[thread_position_in_grid]]) {
    result[index] = inA[index] + inB[index];
}
```

Metal shaders are created in a language known as Metal Shading Language. This is a language designed specifically for GPU Programming, and its syntax is similar to C++. Other GPU shading languages include CUDA and GLSL. We can create shaders in these files. in A shader is basically a GPU Function.

In the above code, we define a shader called `addShader`. It takes 3 arguments, vector A, vector B, and a result vector. Rather than returning from the function, we just have a separate result vector that the GPU will write the results, kind of like modifying a pointer.

Note we have a 4th argument that is called index, this is because the GPU will process every element in the vector at once. The GPU will execute this function x times, where x is the length of the vector. So the function only has to add 2 numbers together, the GPU will automatically loop over the array. This is an important idea in GPU programming and parallel programming in general.

Now, we need to create an Objective-C file that can load the shader and communicate with the GPU.

Create a file called `MetalAdder.h`. The idea is that we will create a class called `MetalAdder` that will handle all the CPU/GPU setup and communication. Our main process can just send in the data.

```objc
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#ifndef MetalAdder_h
#define MetalAdder_h

// Header file for MetalAdder.
@interface MetalAdder : NSObject
// These three methods will be public.
-(instancetype) initWithDevice: (id<MTLDevice>) device;
-(void) prepareData;
-(void) sendComputeCommand;
@end

#endif /* MetalAdder_h */
```

Naturally, we now create the implementation file, `MetalAdder.m`. This file is very complex and I added an explanation at the bottom.

```objc
#import "MetalAdder.h"

// We are going to create a massive array to send into the GPU.
const unsigned int arrayLength = 1 << 24;
const unsigned int bufferSize = arrayLength * sizeof(float);

@implementation MetalAdder
{
    // Initializing any metal variables, like the device, the command queue, and the command buffers.
    // Device will be passed in from main program.
    id<MTLDevice> _mDevice;
    id<MTLComputePipelineState> _mAddFunctionPSO; // The shader will be loaded into here.
    id<MTLCommandQueue> _mCommandQueue; // Initializing the command buffer.
    id<MTLBuffer> _mBufferA; // Our 3 GPU buffers
    id<MTLBuffer> _mBufferB;
    id<MTLBuffer> _mBufferRes;
}

-(instancetype) initWithDevice:(id<MTLDevice>) device
{
    self = [super init];
    if (self) {
        _mDevice = device;
        NSError* error = nil;
        // Find the kernel that we made in addShader.metal
        id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
        id<MTLFunction> addFunction = [defaultLibrary newFunctionWithName:@"addShader"];
        // in addShader.metal, I named the function addFunction
        _mAddFunctionPSO = [_mDevice newComputePipelineStateWithFunction: addFunction error: &error];
        // Initializing a new command queue.
        _mCommandQueue = [_mDevice newCommandQueue];
    }
    return self;
}

-(void) sendComputeCommand {
    // This function performs the actual calculation
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    // adds the buffers into the command buffer and ready it for execution
    [self encodeAddCommand:computeEncoder];
    
    // Execute the shader
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Verify the results once completed.
    [self verifyResults];
    
    
}

-(void) encodeAddCommand:(id<MTLComputeCommandEncoder>)computeEncoder {
    // We are adding the buffers into the command encoder
    [computeEncoder setComputePipelineState:_mAddFunctionPSO];
    [computeEncoder setBuffer:_mBufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferRes offset:0 atIndex:2];
    
    // We are letting the GPU know the size of our array.
    MTLSize gridSize = MTLSizeMake(arrayLength, 1, 1);
    NSUInteger threadGroupSize = _mAddFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > arrayLength) {
        threadGroupSize = arrayLength;
    }
    
    // Informing the GPU of the number of threads it should run with.
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    
}

-(void) prepareData {
    // Initialize the 3 buffers with a fixed length.
    NSLog(@"Arraylength: %d", arrayLength);
    _mBufferA = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _mBufferB = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _mBufferRes = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    
    // Generate the data.
    [self generateRandomFloatData:_mBufferA];
    [self generateRandomFloatData:_mBufferB];
    
    
}

-(void) generateRandomFloatData:(id<MTLBuffer>)buffer {
    // Data generation function.
    float *dataPtr = buffer.contents;
    for (unsigned long i = 0; i < arrayLength; i++) {
        dataPtr[i] = (float)rand()/(float)(RAND_MAX);
    }
}

-(void) verifyResults {
    // Checking results by grabbing the pointers.
    float* A = _mBufferA.contents;
    float* B = _mBufferB.contents;
    float* C = _mBufferRes.contents;
    
    long errors = 0;
    
    for (unsigned long i = 0; i < arrayLength; i++) {
        if (C[i] != (A[i] + B[i])) {
            printf("Compute ERROR: index=%lu result=%g vs %g=a+b\n",
                   i, C[i], A[i] + B[i]);
            errors++;
        }
    }
    NSLog(@"Finished verification");
    NSLog(@"%ld errors found.", errors);
}

@end
```

There is a lot going on in this file.

- We initialize a lot of properties that help us send data into the GPU and receive output.
- `initWithDevice` will setup the GPU, create the command buffer, and load in the shader
- `sendComputeCommand` will just place a command in the command buffer and send it to the GPU
- `encodeAddCommand` does the work of placing the command in the command buffer
- `prepareData` turns the CPU vectors into GPU buffers.
- `generateRandomFloatData` just generates a random vector of x numbers and stores it in memory
- `verifyResults` ensures that the GPU performs the calculation correctly, by performing the same calculation on CPU and then comparing the results.

I mentioned a lot about commands and command buffers. A command is basically an instruction for the GPU, in this case, the command is to tell the GPU to add the two vectors. A command will usually take a shader function and some data. A command buffer is a queue for the commands. The CPU places the commands into the command buffer and sends it off to the GPU all at once. 

I also mentioned that the CPU vectors were turned into GPU Buffers. A buffer is like a GPU variable. It stores some type of data, generally vectors, matrices, etc.

Now, it is time to add some code into `main.m` so that we can run this example. Enter `main.m` and add the following code.

```objc
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "MetalAdder.h"


int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSLog(@"Program Started");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        
        MetalAdder* madder = [[MetalAdder alloc] initWithDevice:device];
        
        [madder prepareData];
        NSLog(@"Data prepared");
        [madder sendComputeCommand];
        
        NSLog(@"Program Completed");
    }
    return 0;
}
```

This file is very simple. It just creates a GPU device and an instance of `MetalAdder`. It runs `prepareData` to generate the vectors and convert them into GPU buffers. Then it runs `sendComputeCommand`, which will run the actual computation on the GPU, and verify the results.

```
Finished verification
0 errors found.
Program Completed
```

If the program is successful, the last 3 lines of output should look like the block above. That's it for this tutorial and I will see you next time.