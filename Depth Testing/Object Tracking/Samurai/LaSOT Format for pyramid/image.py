from PIL import Image

image_path = "Test_Images_Pyramid/Pyramid_test_pic_1.png"
with Image.open(image_path) as img:
    width, height = img.size

print(f"Resolution: {width}x{height}")
