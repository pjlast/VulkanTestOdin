# Detect operating system
UNAME_S := $(shell uname -s)

# Common flags
CFLAGS = -std=c++17 -O2

# OS-specific configuration
ifeq ($(UNAME_S),Darwin)
    # macOS configuration
    BREW_PREFIX := $(shell brew --prefix)
    VULKAN_SDK_PREFIX := /Users/pjlast/VulkanSDK/1.4.304.0/macOS
    
    # Vulkan SDK configuration for macOS
    export VULKAN_SDK := $(VULKAN_SDK_PREFIX)/share/vulkan
    export VK_ICD_FILENAMES := $(VULKAN_SDK)/icd.d/MoltenVK_icd.json
    export VK_LAYER_PATH := $(VULKAN_SDK)/explicit_layer.d
    export DYLD_LIBRARY_PATH := $(BREW_PREFIX)/Cellar/vulkan-validationlayers/1.4.305/lib:$(VULKAN_SDK_PREFIX)/lib
    # export VK_LOADER_DEBUG=all
else
    # Vulkan SDK configuration for Linux (assuming standard installation)
    export VULKAN_SDK := /usr/share/vulkan
    export VK_LAYER_PATH := $(VULKAN_SDK)/explicit_layer.d
    # export VK_LOADER_DEBUG=all
endif

# Build target
VulkanTestOdin: main.odin
	odin build . -debug

# Phony targets
.PHONY: test clean

test: VulkanTestOdin
	./VulkanTestOdin

debug: VulkanTestOdin
	gdb ./VulkanTestOdin

clean:
	rm -f VulkanTestOdin
