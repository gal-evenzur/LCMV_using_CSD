"""
Test script to verify fillline and plotcube functions work correctly.
"""
import numpy as np
import matplotlib.pyplot as plt
from dataset_funcs import fillline, plotcube
import os

tests = {
    'filline': 0,
    'plotcube': 0,
    'create_locations_18_dynamic': 1,
}

# =============================================================================
# Test fillline function
# =============================================================================
print("=" * 50)
print("Testing fillline function")
print("=" * 50)

# Test 1: Regular diagonal line
print("\n1. Diagonal line from (0,0) to (10,5):")
xx, yy = fillline([0, 0], [10, 5], 11)
print(f"   xx = {xx}")
print(f"   yy = {yy}")

# Test 2: Horizontal line (gradient = 0)
print("\n2. Horizontal line from (0,3) to (10,3):")
xx, yy = fillline([0, 3], [10, 3], 6)
print(f"   xx = {xx}")
print(f"   yy = {yy}")

# Test 3: Vertical line (gradient = inf)
print("\n3. Vertical line from (5,0) to (5,10):")
xx, yy = fillline([5, 0], [5, 10], 6)
print(f"   xx = {xx}")
print(f"   yy = {yy}")

# Test 4: Negative slope line
print("\n4. Negative slope line from (0,10) to (10,0):")
xx, yy = fillline([0, 10], [10, 0], 6)
print(f"   xx = {xx}")
print(f"   yy = {yy}")


# =============================================================================
# Visual test: Plot all line types
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 2D lines
ax1 = axes[0]
ax1.set_title("fillline() - Different Line Types")

# Diagonal
xx, yy = fillline([0, 0], [10, 10], 20)
ax1.plot(xx, yy, 'bo-', label='Diagonal', markersize=4)

# Horizontal
xx, yy = fillline([0, 5], [10, 5], 20)
ax1.plot(xx, yy, 'rs-', label='Horizontal', markersize=4)

# Vertical
xx, yy = fillline([5, 0], [5, 10], 20)
ax1.plot(xx, yy, 'g^-', label='Vertical', markersize=4)

# Negative slope
xx, yy = fillline([0, 8], [8, 0], 20)
ax1.plot(xx, yy, 'm*-', label='Negative slope', markersize=4)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()
ax1.grid(True)
ax1.set_aspect('equal')


# =============================================================================
# Test plotcube function
# =============================================================================
print("\n" + "=" * 50)
print("Testing plotcube function")
print("=" * 50)
print("\nCreating 3D plot with multiple cubes...")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# Remove the flat axes[1] and replace with 3D
axes[1].remove()

# Cube 1: Red cube at origin
plotcube([3, 3, 3], [0, 0, 0], 0.6, [1, 0, 0], ax2)
print("   - Red cube: edges=[3,3,3], origin=[0,0,0]")

# Cube 2: Green cube offset
plotcube([2, 4, 2], [5, 0, 0], 0.6, [0, 1, 0], ax2)
print("   - Green cube: edges=[2,4,2], origin=[5,0,0]")

# Cube 3: Blue cube at different position
plotcube([3, 2, 4], [0, 5, 0], 0.6, [0, 0, 1], ax2)
print("   - Blue cube: edges=[3,2,4], origin=[0,5,0]")

# Cube 4: Yellow cube (semi-transparent room-like)
plotcube([8, 8, 5], [0, 0, 0], 0.1, [1, 1, 0], ax2)
print("   - Yellow cube (room): edges=[8,8,5], origin=[0,0,0], alpha=0.1")

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('plotcube() - Multiple Cubes')

# Set axis limits
ax2.set_xlim([0, 10])
ax2.set_ylim([0, 10])
ax2.set_zlim([0, 6])

plt.tight_layout()
py_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(py_path, 'Results')
os.makedirs(result_path, exist_ok=True)
plt.savefig(os.path.join(result_path, 'test_output.png'), dpi=150)
print("\n✓ Plot saved to 'test_output.png'")

print("\n" + "=" * 50)
print("All tests completed successfully!")
print("=" * 50)
