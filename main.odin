package main

import "base:intrinsics"
import "base:runtime"
import "core:fmt"
import "core:log"
import "core:math"
import "core:strings"
import "vendor:glfw"
import vk "vendor:vulkan"

when ODIN_OS == .Darwin {
	// We need to import the VulkanSDK libraries, but to do that we need the useless
	// import of "system:System.framework" (or anything else really) so that the
	// required extra linker flags are executed.
	@(require, extra_linker_flags = "-rpath /Users/pjlast/VulkanSDK/1.4.304.0/macOS/lib")
	foreign import __ "system:System.framework"
}

MAX_FRAMES_IN_FLIGHT :: 2

glfw_error_callback :: proc "c" (code: i32, description: cstring) {
	context = runtime.default_context()
	log.errorf("glfw: %i: %s", code, description)
}

WIDTH :: 800
HEIGHT :: 600

ENABLE_VALIDATION_LAYERS :: #config(ENABLE_VALIDATION_LAYERS, ODIN_DEBUG)

Vertex :: struct {
	pos:   [2]f32,
	color: [3]f32,
}

//odinfmt: disable
vertices := []Vertex {
	{{ 0.0, -0.5}, {1.0, 0.0, 0.0}},
	{{ 0.5,  0.5}, {0.0, 1.0, 0.0}},
	{{-0.5,  0.5}, {0.0, 0.0, 1.0}},
}
//odinfmt: enable

get_vertex_binding_description :: proc() -> vk.VertexInputBindingDescription {
	binding_description := vk.VertexInputBindingDescription {
		binding   = 0,
		stride    = size_of(Vertex),
		inputRate = .VERTEX,
	}

	return binding_description
}

get_vertex_attribute_descriptions :: proc() -> [2]vk.VertexInputAttributeDescription {
	attribute_descriptions := [2]vk.VertexInputAttributeDescription {
		{binding = 0, location = 0, format = .R32G32_SFLOAT, offset = u32(offset_of(Vertex, pos))},
		{
			binding = 0,
			location = 1,
			format = .R32G32B32_SFLOAT,
			offset = u32(offset_of(Vertex, color)),
		},
	}

	return attribute_descriptions
}

window: glfw.WindowHandle
instance: vk.Instance
debug_messenger: vk.DebugUtilsMessengerEXT
surface: vk.SurfaceKHR
physical_device: vk.PhysicalDevice
device: vk.Device
graphics_queue: vk.Queue
present_queue: vk.Queue
swap_chain: vk.SwapchainKHR
swap_chain_images: [dynamic]vk.Image
swap_chain_image_format: vk.Format
swap_chain_extent: vk.Extent2D
swap_chain_image_views: [dynamic]vk.ImageView
render_pass: vk.RenderPass
pipeline_layout: vk.PipelineLayout
graphics_pipeline: vk.Pipeline
swap_chain_framebuffers: [dynamic]vk.Framebuffer
command_pool: vk.CommandPool
command_buffers: [MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer
image_available_semaphores: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore
render_finished_semaphores: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore
in_flight_fences: [MAX_FRAMES_IN_FLIGHT]vk.Fence
framebuffer_resized: bool
vertex_buffer: vk.Buffer
vertex_buffer_memory: vk.DeviceMemory

device_extensions: [dynamic]cstring

init_window :: proc() {
	glfw.Init()

	// Don't create an OpenGL context
	glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)

	window = glfw.CreateWindow(WIDTH, HEIGHT, "Vulkan", nil, nil)

	glfw.SetFramebufferSizeCallback(window, proc "c" (_: glfw.WindowHandle, _, _: i32) {
		framebuffer_resized = true
	})
}

init_vulkan :: proc() {
	append(&device_extensions, vk.KHR_SWAPCHAIN_EXTENSION_NAME)
	create_instance()
	setup_debug_messenger()
	create_surface()
	pick_physical_device()
	create_logical_device()
	create_swap_chain()
	create_image_views()
	create_render_pass()
	create_graphics_pipeline()
	create_framebuffers()
	create_command_pool()
	create_vertex_buffer()
	create_command_buffers()
	create_sync_objects()
}

find_memory_type :: proc(type_filter: u32, properties: vk.MemoryPropertyFlags) -> u32 {
	mem_properties: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(physical_device, &mem_properties)

	for i: u32 = 0; i < mem_properties.memoryTypeCount; i += 1 {
		if type_filter & (1 << i) > 0 &&
		   (mem_properties.memoryTypes[i].propertyFlags & properties) == properties {
			return i
		}
	}

	log.panic("failed to find suitable memory type")
}

create_vertex_buffer :: proc() {
	buffer_info := vk.BufferCreateInfo {
		sType       = .BUFFER_CREATE_INFO,
		size        = vk.DeviceSize(size_of(vertices[0]) * len(vertices)),
		usage       = {.VERTEX_BUFFER},
		sharingMode = .EXCLUSIVE,
	}

	must(vk.CreateBuffer(device, &buffer_info, nil, &vertex_buffer))

	mem_requirements: vk.MemoryRequirements

	vk.GetBufferMemoryRequirements(device, vertex_buffer, &mem_requirements)

	alloc_info := vk.MemoryAllocateInfo {
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = mem_requirements.size,
		memoryTypeIndex = find_memory_type(
			mem_requirements.memoryTypeBits,
			{.HOST_VISIBLE, .HOST_COHERENT},
		),
	}

	must(vk.AllocateMemory(device, &alloc_info, nil, &vertex_buffer_memory))

	vk.BindBufferMemory(device, vertex_buffer, vertex_buffer_memory, 0)

	data: rawptr
	vk.MapMemory(device, vertex_buffer_memory, 0, buffer_info.size, {}, &data)
	intrinsics.mem_copy_non_overlapping(data, raw_data(vertices), buffer_info.size)
	vk.UnmapMemory(device, vertex_buffer_memory)
}

create_sync_objects :: proc() {
	semaphore_info := vk.SemaphoreCreateInfo {
		sType = .SEMAPHORE_CREATE_INFO,
	}

	fence_info := vk.FenceCreateInfo {
		sType = .FENCE_CREATE_INFO,
		flags = {.SIGNALED}, // Create fence in signaled state so we don't block on first render
	}

	for i := 0; i < MAX_FRAMES_IN_FLIGHT; i += 1 {
		must(vk.CreateSemaphore(device, &semaphore_info, nil, &image_available_semaphores[i]))
		must(vk.CreateSemaphore(device, &semaphore_info, nil, &render_finished_semaphores[i]))
		must(vk.CreateFence(device, &fence_info, nil, &in_flight_fences[i]))
	}
}

create_command_buffers :: proc() {
	alloc_info := vk.CommandBufferAllocateInfo {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = command_pool,
		level              = .PRIMARY,
		commandBufferCount = MAX_FRAMES_IN_FLIGHT,
	}

	must(vk.AllocateCommandBuffers(device, &alloc_info, raw_data(command_buffers[:])))
}

create_command_pool :: proc() {
	queue_family_indices := find_queue_families(physical_device)

	pool_info := vk.CommandPoolCreateInfo {
		sType            = .COMMAND_POOL_CREATE_INFO,
		flags            = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = queue_family_indices.graphics_family.(u32),
	}

	must(vk.CreateCommandPool(device, &pool_info, nil, &command_pool))
}

create_framebuffers :: proc() {
	resize(&swap_chain_framebuffers, len(swap_chain_image_views))

	for swap_chain_image_view, i in swap_chain_image_views {
		attachments := []vk.ImageView{swap_chain_image_view}

		framebuffer_info := vk.FramebufferCreateInfo {
			sType           = .FRAMEBUFFER_CREATE_INFO,
			renderPass      = render_pass,
			attachmentCount = 1,
			pAttachments    = raw_data(attachments),
			width           = swap_chain_extent.width,
			height          = swap_chain_extent.height,
			layers          = 1,
		}

		must(vk.CreateFramebuffer(device, &framebuffer_info, nil, &swap_chain_framebuffers[i]))
	}
}

create_shader_module :: proc(code: []u8) -> vk.ShaderModule {
	create_info := vk.ShaderModuleCreateInfo {
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(code),
		pCode    = (^u32)(raw_data(code)),
	}

	shader_module: vk.ShaderModule
	must(vk.CreateShaderModule(device, &create_info, nil, &shader_module))

	return shader_module
}

create_graphics_pipeline :: proc() {
	vert_shader_code :: #load("shaders/vert.spv")
	frag_shader_code :: #load("shaders/frag.spv")

	vert_shader_module := create_shader_module(vert_shader_code)
	frag_shader_module := create_shader_module(frag_shader_code)

	vert_shader_stage_info := vk.PipelineShaderStageCreateInfo {
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = {.VERTEX},
		module = vert_shader_module,
		pName  = "main",
	}

	frag_shader_stage_info := vk.PipelineShaderStageCreateInfo {
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = {.FRAGMENT},
		module = frag_shader_module,
		pName  = "main",
	}

	shader_stages := []vk.PipelineShaderStageCreateInfo {
		vert_shader_stage_info,
		frag_shader_stage_info,
	}

	dynamic_states := []vk.DynamicState{.VIEWPORT, .SCISSOR}

	dynamic_state := vk.PipelineDynamicStateCreateInfo {
		sType             = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		dynamicStateCount = u32(len(dynamic_states)),
		pDynamicStates    = raw_data(dynamic_states),
	}

	binding_description := get_vertex_binding_description()
	attribute_descriptions := get_vertex_attribute_descriptions()

	vertex_input_info := vk.PipelineVertexInputStateCreateInfo {
		sType                           = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		vertexBindingDescriptionCount   = 1,
		pVertexBindingDescriptions      = &binding_description,
		vertexAttributeDescriptionCount = u32(len(attribute_descriptions)),
		pVertexAttributeDescriptions    = raw_data(attribute_descriptions[:]),
	}

	input_assembly := vk.PipelineInputAssemblyStateCreateInfo {
		sType                  = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		topology               = .TRIANGLE_LIST,
		primitiveRestartEnable = false,
	}

	viewport := vk.Viewport {
		x        = 0.0,
		y        = 0.0,
		width    = f32(swap_chain_extent.width),
		height   = f32(swap_chain_extent.height),
		minDepth = 0.0,
		maxDepth = 1.0,
	}

	scissor := vk.Rect2D {
		offset = {0, 0},
		extent = swap_chain_extent,
	}

	viewport_state := vk.PipelineViewportStateCreateInfo {
		sType         = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		viewportCount = 1,
		pViewports    = &viewport,
		scissorCount  = 1,
		pScissors     = &scissor,
	}

	rasterizer := vk.PipelineRasterizationStateCreateInfo {
		sType                   = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		depthClampEnable        = false,
		rasterizerDiscardEnable = false,
		polygonMode             = .FILL,
		cullMode                = {.BACK},
		frontFace               = .CLOCKWISE,
		depthBiasEnable         = false,
		lineWidth               = 1.0,
	}

	multisampling := vk.PipelineMultisampleStateCreateInfo {
		sType                 = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		rasterizationSamples  = {._1},
		sampleShadingEnable   = false,
		minSampleShading      = 1.0,
		pSampleMask           = nil,
		alphaToCoverageEnable = false,
		alphaToOneEnable      = false,
	}

	color_blend_attachment := vk.PipelineColorBlendAttachmentState {
		blendEnable         = false,
		srcColorBlendFactor = .ONE,
		dstColorBlendFactor = .ZERO,
		colorBlendOp        = .ADD,
		srcAlphaBlendFactor = .ONE,
		dstAlphaBlendFactor = .ZERO,
		alphaBlendOp        = .ADD,
		colorWriteMask      = {.R, .G, .B, .A},
	}

	color_blending := vk.PipelineColorBlendStateCreateInfo {
		sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		logicOpEnable   = false,
		logicOp         = .COPY,
		attachmentCount = 1,
		pAttachments    = &color_blend_attachment,
		blendConstants  = {0.0, 0.0, 0.0, 0.0},
	}

	pipeline_layout_info := vk.PipelineLayoutCreateInfo {
		sType = .PIPELINE_LAYOUT_CREATE_INFO,
	}

	must(vk.CreatePipelineLayout(device, &pipeline_layout_info, nil, &pipeline_layout))

	pipeline_info := vk.GraphicsPipelineCreateInfo {
		sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
		stageCount          = 2,
		pStages             = raw_data(shader_stages),
		pVertexInputState   = &vertex_input_info,
		pInputAssemblyState = &input_assembly,
		pViewportState      = &viewport_state,
		pRasterizationState = &rasterizer,
		pMultisampleState   = &multisampling,
		pDepthStencilState  = nil,
		pColorBlendState    = &color_blending,
		pDynamicState       = &dynamic_state,
		layout              = pipeline_layout,
		renderPass          = render_pass,
		subpass             = 0,
		basePipelineIndex   = -1,
	}

	must(vk.CreateGraphicsPipelines(device, 0, 1, &pipeline_info, nil, &graphics_pipeline))

	vk.DestroyShaderModule(device, frag_shader_module, nil)
	vk.DestroyShaderModule(device, vert_shader_module, nil)
}

create_render_pass :: proc() {
	color_attachment := vk.AttachmentDescription {
		format         = swap_chain_image_format,
		samples        = {._1},
		loadOp         = .CLEAR,
		storeOp        = .STORE,
		stencilLoadOp  = .DONT_CARE,
		stencilStoreOp = .DONT_CARE,
		initialLayout  = .UNDEFINED,
		finalLayout    = .PRESENT_SRC_KHR,
	}

	color_attachment_ref := vk.AttachmentReference {
		attachment = 0,
		layout     = .COLOR_ATTACHMENT_OPTIMAL,
	}

	subpass := vk.SubpassDescription {
		pipelineBindPoint    = .GRAPHICS,
		colorAttachmentCount = 1,
		pColorAttachments    = &color_attachment_ref,
	}

	dependency := vk.SubpassDependency {
		srcSubpass    = vk.SUBPASS_EXTERNAL,
		dstSubpass    = 0,
		srcStageMask  = {.COLOR_ATTACHMENT_OUTPUT},
		dstStageMask  = {.COLOR_ATTACHMENT_OUTPUT},
		srcAccessMask = {},
		dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
	}

	render_pass_info := vk.RenderPassCreateInfo {
		sType           = .RENDER_PASS_CREATE_INFO,
		attachmentCount = 1,
		pAttachments    = &color_attachment,
		subpassCount    = 1,
		pSubpasses      = &subpass,
		dependencyCount = 1,
		pDependencies   = &dependency,
	}

	must(vk.CreateRenderPass(device, &render_pass_info, nil, &render_pass))
}

create_image_views :: proc() {
	resize(&swap_chain_image_views, len(swap_chain_images))

	for swap_chain_image, i in swap_chain_images {
		create_info := vk.ImageViewCreateInfo {
			sType = .IMAGE_VIEW_CREATE_INFO,
			image = swap_chain_image,
			viewType = .D2,
			format = swap_chain_image_format,
			components = {r = .IDENTITY, g = .IDENTITY, b = .IDENTITY, a = .IDENTITY},
			subresourceRange = {
				aspectMask = {.COLOR},
				baseMipLevel = 0,
				levelCount = 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
		}

		must(vk.CreateImageView(device, &create_info, nil, &swap_chain_image_views[i]))
	}
}

choose_swap_surface_format :: proc(
	available_formats: []vk.SurfaceFormatKHR,
) -> vk.SurfaceFormatKHR {
	for available_format in available_formats {
		if available_format.format == .B8G8R8A8_SRGB &&
		   available_format.colorSpace == .SRGB_NONLINEAR {
			return available_format
		}
	}

	return available_formats[0]
}

choose_swap_present_mode :: proc(
	available_present_modes: []vk.PresentModeKHR,
) -> vk.PresentModeKHR {
	for available_present_mode in available_present_modes {
		if available_present_mode == .MAILBOX {
			return available_present_mode
		}
	}

	return .FIFO
}

choose_swap_extent :: proc(capabilities: vk.SurfaceCapabilitiesKHR) -> vk.Extent2D {
	if capabilities.currentExtent.width != max(u32) {
		return capabilities.currentExtent
	}

	width, height := glfw.GetFramebufferSize(window)

	actual_extent := vk.Extent2D{u32(width), u32(height)}

	actual_extent.width = math.clamp(
		actual_extent.width,
		capabilities.minImageExtent.width,
		capabilities.maxImageExtent.width,
	)
	actual_extent.height = math.clamp(
		actual_extent.height,
		capabilities.minImageExtent.height,
		capabilities.maxImageExtent.height,
	)

	return actual_extent
}

create_swap_chain :: proc() {
	swap_chain_support := query_swap_chain_support(physical_device)

	surface_format := choose_swap_surface_format(swap_chain_support.formats[:])
	present_mode := choose_swap_present_mode(swap_chain_support.present_modes[:])
	extent := choose_swap_extent(swap_chain_support.capabilities)

	image_count := swap_chain_support.capabilities.minImageCount + 1

	if swap_chain_support.capabilities.maxImageCount > 0 &&
	   image_count > swap_chain_support.capabilities.maxImageCount {
		image_count = swap_chain_support.capabilities.maxImageCount
	}

	create_info := vk.SwapchainCreateInfoKHR {
		sType            = .SWAPCHAIN_CREATE_INFO_KHR,
		surface          = surface,
		minImageCount    = image_count,
		imageFormat      = surface_format.format,
		imageColorSpace  = surface_format.colorSpace,
		imageExtent      = extent,
		imageArrayLayers = 1,
		imageUsage       = {.COLOR_ATTACHMENT},
	}

	indices := find_queue_families(physical_device)
	queue_family_indices := []u32{indices.graphics_family.(u32), indices.present_family.(u32)}

	if indices.graphics_family != indices.present_family {
		create_info.imageSharingMode = .CONCURRENT
		create_info.queueFamilyIndexCount = 2
		create_info.pQueueFamilyIndices = raw_data(queue_family_indices)
	} else {
		create_info.imageSharingMode = .EXCLUSIVE
	}

	create_info.preTransform = swap_chain_support.capabilities.currentTransform

	create_info.compositeAlpha = {.OPAQUE}
	create_info.presentMode = present_mode
	create_info.clipped = true

	must(vk.CreateSwapchainKHR(device, &create_info, nil, &swap_chain))

	vk.GetSwapchainImagesKHR(device, swap_chain, &image_count, nil)
	resize(&swap_chain_images, image_count)
	vk.GetSwapchainImagesKHR(device, swap_chain, &image_count, raw_data(swap_chain_images))

	swap_chain_image_format = surface_format.format
	swap_chain_extent = extent
}

recreate_swap_chain :: proc() {
	width, height := glfw.GetFramebufferSize(window)
	for width == 0 || height == 0 {
		width, height = glfw.GetFramebufferSize(window)
		glfw.WaitEvents()
	}

	vk.DeviceWaitIdle(device)

	cleanup_swap_chain()

	create_swap_chain()
	create_image_views()
	create_framebuffers()
}

cleanup_swap_chain :: proc() {
	for framebuffer in swap_chain_framebuffers {
		vk.DestroyFramebuffer(device, framebuffer, nil)
	}

	for image_view in swap_chain_image_views {
		vk.DestroyImageView(device, image_view, nil)
	}

	vk.DestroySwapchainKHR(device, swap_chain, nil)
}

create_logical_device :: proc() {
	indices := find_queue_families(physical_device)
	queue_create_infos := make([dynamic]vk.DeviceQueueCreateInfo, context.temp_allocator)
	unique_queue_families := make(map[u32]struct {}, context.temp_allocator)
	unique_queue_families[indices.graphics_family.(u32)] = {}
	unique_queue_families[indices.present_family.(u32)] = {}

	queue_priority: f32 = 1.0

	for queue_family in unique_queue_families {
		queue_create_info := vk.DeviceQueueCreateInfo {
			sType            = .DEVICE_QUEUE_CREATE_INFO,
			queueFamilyIndex = indices.graphics_family.(u32),
			queueCount       = 1,
			pQueuePriorities = &queue_priority,
		}
		append(&queue_create_infos, queue_create_info)
	}

	device_features: vk.PhysicalDeviceFeatures

	create_info := vk.DeviceCreateInfo {
		sType                = .DEVICE_CREATE_INFO,
		pQueueCreateInfos    = raw_data(queue_create_infos),
		queueCreateInfoCount = u32(len(queue_create_infos)),
		pEnabledFeatures     = &device_features,
	}

	when ODIN_OS == .Darwin {
		append(&device_extensions, "VK_KHR_portability_subset")
	}

	create_info.enabledExtensionCount = u32(len(device_extensions))
	create_info.ppEnabledExtensionNames = raw_data(device_extensions)

	when ENABLE_VALIDATION_LAYERS {
		create_info.enabledLayerCount = u32(len(validation_layers))
		create_info.ppEnabledLayerNames = raw_data(validation_layers)
	} else {
		create_info.enabledLayerCount = 0
	}

	must(vk.CreateDevice(physical_device, &create_info, nil, &device))

	vk.GetDeviceQueue(device, indices.graphics_family.(u32), 0, &graphics_queue)
	vk.GetDeviceQueue(device, indices.present_family.(u32), 0, &present_queue)
}

Queue_Family_Indices :: struct {
	graphics_family: Maybe(u32),
	present_family:  Maybe(u32),
}

is_indices_complete :: proc(indices: Queue_Family_Indices) -> bool {
	return indices.graphics_family != nil && indices.present_family != nil
}

find_queue_families :: proc(device: vk.PhysicalDevice) -> Queue_Family_Indices {
	queue_family_count: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nil)

	queue_families := make([]vk.QueueFamilyProperties, queue_family_count, context.temp_allocator)
	vk.GetPhysicalDeviceQueueFamilyProperties(
		device,
		&queue_family_count,
		raw_data(queue_families),
	)

	indices: Queue_Family_Indices

	for queue_family, i in queue_families {
		if vk.QueueFlag.GRAPHICS in queue_family.queueFlags {
			indices.graphics_family = u32(i)
		}

		present_support: b32 = false
		vk.GetPhysicalDeviceSurfaceSupportKHR(device, u32(i), surface, &present_support)

		if present_support {
			indices.present_family = u32(i)
		}

		if is_indices_complete(indices) {
			break
		}
	}

	return indices
}

check_device_extension_support :: proc(device: vk.PhysicalDevice) -> bool {
	extension_count: u32
	vk.EnumerateDeviceExtensionProperties(device, nil, &extension_count, nil)

	available_extensions := make([]vk.ExtensionProperties, extension_count, context.temp_allocator)
	vk.EnumerateDeviceExtensionProperties(
		device,
		nil,
		&extension_count,
		raw_data(available_extensions),
	)

	required_extensions := make(map[cstring]struct {}, context.temp_allocator)
	for device_extension in device_extensions {
		required_extensions[device_extension] = {}
	}

	for &extension in available_extensions {
		delete_key(&required_extensions, byte_arr_cstr(&extension.extensionName))
	}

	return len(required_extensions) == 0
}

Swap_Chain_Support_Details :: struct {
	capabilities:  vk.SurfaceCapabilitiesKHR,
	formats:       [dynamic]vk.SurfaceFormatKHR,
	present_modes: [dynamic]vk.PresentModeKHR,
}

query_swap_chain_support :: proc(device: vk.PhysicalDevice) -> Swap_Chain_Support_Details {
	details: Swap_Chain_Support_Details

	vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities)

	format_count: u32
	vk.GetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nil)

	if format_count != 0 {
		resize(&details.formats, format_count)
		vk.GetPhysicalDeviceSurfaceFormatsKHR(
			device,
			surface,
			&format_count,
			raw_data(details.formats),
		)
	}

	present_mode_count: u32
	vk.GetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nil)

	if present_mode_count != 0 {
		resize(&details.present_modes, present_mode_count)
		vk.GetPhysicalDeviceSurfacePresentModesKHR(
			device,
			surface,
			&present_mode_count,
			raw_data(details.present_modes),
		)
	}

	return details
}

is_device_suitable :: proc(device: vk.PhysicalDevice) -> bool {
	indices := find_queue_families(device)

	extensions_supported := check_device_extension_support(device)

	swap_chain_adequate := false
	if extensions_supported {
		swap_chain_support := query_swap_chain_support(device)
		swap_chain_adequate =
			len(swap_chain_support.formats) != 0 && len(swap_chain_support.present_modes) != 0
	}

	return is_indices_complete(indices) && extensions_supported && swap_chain_adequate
}

pick_physical_device :: proc() {
	device_count: u32
	vk.EnumeratePhysicalDevices(instance, &device_count, nil)

	assert(device_count != 0, "failed to find GPUs with vulkan support")

	devices := make([]vk.PhysicalDevice, device_count, context.temp_allocator)

	vk.EnumeratePhysicalDevices(instance, &device_count, raw_data(devices))

	for device in devices {
		if is_device_suitable(device) {
			physical_device = device
			break
		}
	}

	if physical_device == nil {
		log.panicf("failed to find suitable GPU")
	}
}

create_surface :: proc() {
	must(glfw.CreateWindowSurface(instance, window, nil, &surface))
}

setup_debug_messenger :: proc() {
	when ENABLE_VALIDATION_LAYERS {
		create_info: vk.DebugUtilsMessengerCreateInfoEXT
		populate_debug_messenger_create_info(&create_info)
		must(vk.CreateDebugUtilsMessengerEXT(instance, &create_info, nil, &debug_messenger))
	}
}

validation_layers := []cstring{"VK_LAYER_KHRONOS_validation"}

check_validation_layer_support :: proc() -> bool {
	layer_count: u32
	vk.EnumerateInstanceLayerProperties(&layer_count, nil)

	available_layers := make([]vk.LayerProperties, layer_count, context.temp_allocator)
	vk.EnumerateInstanceLayerProperties(&layer_count, raw_data(available_layers))

	for layer_name in validation_layers {
		layer_found := false

		for &layer_properties in available_layers {
			if string(layer_name) == byte_arr_str(&layer_properties.layerName) {
				layer_found = true
				break
			}
		}

		if !layer_found {
			return false
		}
	}

	return true
}

get_required_extensions :: proc() -> [dynamic]cstring {
	required_glfw_extensions := glfw.GetRequiredInstanceExtensions()
	extensions := make([dynamic]cstring, 0, len(required_glfw_extensions))

	append(&extensions, ..required_glfw_extensions)

	when ENABLE_VALIDATION_LAYERS {
		append(&extensions, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)
	}

	when ODIN_OS == .Darwin {
		append(
			&extensions,
			vk.KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
			vk.KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
		)
	}

	return extensions
}

debug_callback :: proc "system" (
	messageSeverity: vk.DebugUtilsMessageSeverityFlagsEXT,
	messageType: vk.DebugUtilsMessageTypeFlagsEXT,
	pCallbackData: ^vk.DebugUtilsMessengerCallbackDataEXT,
	pUserData: rawptr,
) -> b32 {
	context = runtime.default_context()

	fmt.println("validation layer: ", pCallbackData.pMessage)

	return false
}

populate_debug_messenger_create_info :: proc(create_info: ^vk.DebugUtilsMessengerCreateInfoEXT) {
	create_info.sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT
	create_info.messageSeverity = {.VERBOSE, .WARNING, .ERROR}
	create_info.messageType = {.GENERAL, .VALIDATION, .PERFORMANCE}
	create_info.pfnUserCallback = debug_callback
}

create_instance :: proc() {
	when ENABLE_VALIDATION_LAYERS {
		if !check_validation_layer_support() {
			log.panicf("validation layers requested, but not available")
		}
	}

	app_info := vk.ApplicationInfo {
		sType              = .APPLICATION_INFO,
		pApplicationName   = "Hello Triangle",
		applicationVersion = vk.MAKE_VERSION(1, 0, 0),
		pEngineName        = "No Engine",
		engineVersion      = vk.MAKE_VERSION(1, 0, 0),
		apiVersion         = vk.API_VERSION_1_0,
	}

	create_info := vk.InstanceCreateInfo {
		sType            = .INSTANCE_CREATE_INFO,
		pApplicationInfo = &app_info,
	}

	extensions := get_required_extensions()
	defer delete(extensions)

	when ODIN_OS == .Darwin {
		create_info.flags |= {.ENUMERATE_PORTABILITY_KHR}
	}

	create_info.enabledExtensionCount = u32(len(extensions))
	create_info.ppEnabledExtensionNames = raw_data(extensions)

	when ENABLE_VALIDATION_LAYERS {
		create_info.enabledLayerCount = u32(len(validation_layers))
		create_info.ppEnabledLayerNames = raw_data(validation_layers)

		debug_create_info: vk.DebugUtilsMessengerCreateInfoEXT
		populate_debug_messenger_create_info(&debug_create_info)
		create_info.pNext = &debug_create_info
	} else {
		create_info.enabledLayerCount = 0
		create_info.pNext = nil
	}

	must(vk.CreateInstance(&create_info, nil, &instance))

	vk.load_proc_addresses_instance(instance)
	assert(vk.DestroyInstance != nil, "vulkan instance function pointers not loaded")
}

record_command_buffer :: proc(command_buffer: vk.CommandBuffer, image_index: u32) {
	begin_info := vk.CommandBufferBeginInfo {
		sType            = .COMMAND_BUFFER_BEGIN_INFO,
		flags            = {},
		pInheritanceInfo = nil,
	}

	must(vk.BeginCommandBuffer(command_buffer, &begin_info))

	render_pass_info := vk.RenderPassBeginInfo {
		sType = .RENDER_PASS_BEGIN_INFO,
		renderPass = render_pass,
		framebuffer = swap_chain_framebuffers[image_index],
		renderArea = {offset = {0, 0}, extent = swap_chain_extent},
	}

	clear_color := vk.ClearValue {
		color = {float32 = {0.0, 0.0, 0.0, 1.0}},
	}
	render_pass_info.clearValueCount = 1
	render_pass_info.pClearValues = &clear_color

	vk.CmdBeginRenderPass(command_buffer, &render_pass_info, .INLINE)
	vk.CmdBindPipeline(command_buffer, .GRAPHICS, graphics_pipeline)

	vertex_buffers := []vk.Buffer{vertex_buffer}
	offsets := []vk.DeviceSize{0}
	vk.CmdBindVertexBuffers(command_buffer, 0, 1, raw_data(vertex_buffers), raw_data(offsets))

	viewport := vk.Viewport {
		x        = 0.0,
		y        = 0.0,
		width    = f32(swap_chain_extent.width),
		height   = f32(swap_chain_extent.height),
		minDepth = 0.0,
		maxDepth = 1.0,
	}

	vk.CmdSetViewport(command_buffer, 0, 1, &viewport)

	scissor := vk.Rect2D {
		offset = {0, 0},
		extent = swap_chain_extent,
	}

	vk.CmdSetScissor(command_buffer, 0, 1, &scissor)

	vk.CmdDraw(command_buffer, u32(len(vertices)), 1, 0, 0)

	vk.CmdEndRenderPass(command_buffer)

	must(vk.EndCommandBuffer(command_buffer))
}

draw_frame :: proc(current_frame: u32) -> u32 {
	vk.WaitForFences(device, 1, &in_flight_fences[current_frame], true, max(u64))

	image_index: u32
	result := vk.AcquireNextImageKHR(
		device,
		swap_chain,
		max(u64),
		image_available_semaphores[current_frame],
		0,
		&image_index,
	)

	#partial switch result {
	case .ERROR_OUT_OF_DATE_KHR:
		{
			recreate_swap_chain()
			return current_frame
		}
	case .SUCCESS, .SUBOPTIMAL_KHR:
		vk.ResetFences(device, 1, &in_flight_fences[current_frame])
	case:
		log.panic("failed to acquire swap chain image")
	}

	vk.ResetCommandBuffer(command_buffers[current_frame], {})
	record_command_buffer(command_buffers[current_frame], image_index)

	wait_semaphores := []vk.Semaphore{image_available_semaphores[current_frame]}
	wait_stages := []vk.PipelineStageFlags{{.COLOR_ATTACHMENT_OUTPUT}}

	submit_info := vk.SubmitInfo {
		sType                = .SUBMIT_INFO,
		waitSemaphoreCount   = 1,
		pWaitSemaphores      = raw_data(wait_semaphores),
		pWaitDstStageMask    = raw_data(wait_stages),
		commandBufferCount   = 1,
		pCommandBuffers      = &command_buffers[current_frame],
		signalSemaphoreCount = 1,
		pSignalSemaphores    = &render_finished_semaphores[current_frame],
	}

	must(vk.QueueSubmit(graphics_queue, 1, &submit_info, in_flight_fences[current_frame]))

	swapchains := []vk.SwapchainKHR{swap_chain}
	present_info := vk.PresentInfoKHR {
		sType              = .PRESENT_INFO_KHR,
		waitSemaphoreCount = 1,
		pWaitSemaphores    = &render_finished_semaphores[current_frame],
		swapchainCount     = 1,
		pSwapchains        = raw_data(swapchains),
		pImageIndices      = &image_index,
	}

	result = vk.QueuePresentKHR(present_queue, &present_info)

	if result == .ERROR_OUT_OF_DATE_KHR || result == .SUBOPTIMAL_KHR || framebuffer_resized {
		framebuffer_resized = false
		recreate_swap_chain()
	} else if result != .SUCCESS {
		log.panic("failed to present swap chain image")
	}

	return (current_frame + 1) & MAX_FRAMES_IN_FLIGHT
}

main_loop :: proc() {
	current_frame: u32 = 0
	for !glfw.WindowShouldClose(window) {
		free_all(context.temp_allocator)

		glfw.PollEvents()
		current_frame = draw_frame(current_frame)
	}

	vk.DeviceWaitIdle(device)
}

cleanup :: proc() {
	for i := 0; i < MAX_FRAMES_IN_FLIGHT; i += 1 {
		vk.DestroySemaphore(device, image_available_semaphores[i], nil)
		vk.DestroySemaphore(device, render_finished_semaphores[i], nil)
		vk.DestroyFence(device, in_flight_fences[i], nil)
	}

	vk.DestroyCommandPool(device, command_pool, nil)

	cleanup_swap_chain()
	delete(swap_chain_framebuffers)
	delete(swap_chain_image_views)
	delete(swap_chain_images)

	vk.DestroyBuffer(device, vertex_buffer, nil)
	vk.FreeMemory(device, vertex_buffer_memory, nil)

	vk.DestroyPipeline(device, graphics_pipeline, nil)
	vk.DestroyPipelineLayout(device, pipeline_layout, nil)
	vk.DestroyRenderPass(device, render_pass, nil)

	vk.DestroyDevice(device, nil)
	vk.DestroySurfaceKHR(instance, surface, nil)
	when ENABLE_VALIDATION_LAYERS {
		vk.DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nil)
	}
	vk.DestroyInstance(instance, nil)
	delete(device_extensions)

	glfw.DestroyWindow(window)
	glfw.Terminate()
}

main :: proc() {
	glfw.SetErrorCallback(glfw_error_callback)

	init_window()

	vk.load_proc_addresses(rawptr(glfw.GetInstanceProcAddress))
	assert(vk.CreateInstance != nil, "vulkan function pointers not loaded")

	init_vulkan()
	main_loop()
	cleanup()
}

byte_arr_str :: proc(arr: ^[$N]byte) -> string {
	return strings.truncate_to_byte(string(arr[:]), 0)
}

byte_arr_cstr :: proc(arr: ^[$N]byte) -> cstring {
	return strings.clone_to_cstring(strings.truncate_to_byte(string(arr[:]), 0))
}

must :: proc(result: vk.Result, loc := #caller_location) {
	if result != .SUCCESS {
		log.panicf("vulkan failure %v", result, location = loc)
	}
}
