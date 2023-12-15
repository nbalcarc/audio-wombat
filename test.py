from birds import audio_wombat, audio_wombat1, cu_maxmul
import numpy as np
import unittest
from time import perf_counter

frame = np.array([[0, 0, 0, 2, 3, 0]], dtype = np.uint8)
image = np.array([3, 4, 0, 0, 0, 0], dtype = np.uint8)
buff = np.array([0], dtype = np.uint32)

frame_pointer = frame.ctypes.data
image_pointer = image.ctypes.data
buff_pointer = buff.ctypes.data

audio_wombat(frame_pointer, image_pointer, buff_pointer, 6, 1, 1)
print(buff[0])




# fn audio_wombat(frames_pointer: usize, image_pointer: usize, output_pointer: usize, image_size: usize, rotation_amount: usize, frame_count: usize) {
    
def audio_wombat_test_helper(f, i, r) -> list[int]:
    if isinstance(f, list):
        f = np.array(f, dtype = np.uint8)
    i = np.array(i, dtype = np.uint8)
    out = np.zeros(f.shape[0], dtype = np.uint32)
    f_pointer = f.ctypes.data
    i_pointer = i.ctypes.data
    o_pointer = out.ctypes.data
    audio_wombat(f_pointer, i_pointer, o_pointer, f.shape[1], r, f.shape[0])
    return list(out)

def audio_wombat_test_helper1(f, i, r) -> list[int]:
    if isinstance(f, list):
        f = np.array(f, dtype = np.uint8)
    i = np.array(i, dtype = np.uint8)
    out = np.zeros(f.shape[0], dtype = np.uint32)
    f_pointer = f.ctypes.data
    i_pointer = i.ctypes.data
    o_pointer = out.ctypes.data
    audio_wombat1(f_pointer, i_pointer, o_pointer, f.shape[1], r, f.shape[0])
    return list(out)

class TestAudioWombat(unittest.TestCase):
    def test_basic(self):
        out = audio_wombat_test_helper1([[1,2,3,4,5,6,7,8]], [2,3,4,5,6,7,8,1], 1)[0]
        self.assertEqual(out, 0)

    def test_basic_2(self):
        out = audio_wombat_test_helper1([[1,2,1,2,1,2,1,2]], [2,1,2,1,2,1,2,1], 1)[0]
        self.assertEqual(out, 0)

    def test_basic_3(self):
        out = audio_wombat_test_helper1([[1,3,1,2,1,3,1,2]], [2,1,2,1,2,1,2,1], 1)[0]
        self.assertEqual(out, 2)

    def test_basic_4(self):
        out = audio_wombat_test_helper1([[99]], [1], 1)[0]
        self.assertEqual(out, 98)

    def test_basic_5(self):
        out = audio_wombat_test_helper1([[1]], [99], 1)[0]
        self.assertEqual(out, 98)

    def test_basic_6(self):
        out = audio_wombat_test_helper1([[1,2,3,4,5]], [5,4,3,2,1], 1)[0]
        self.assertEqual(out, 6)

    def test_multiple_frames(self):
        out = audio_wombat_test_helper1([[1,2,3,4,5], [0, 0, 10, 0, 0]], [5,4,3,2,1], 1)
        self.assertEqual(out, [6, 5])

    def test_multiple_frames_1(self):
        out = audio_wombat_test_helper1([[1,1,1,1,1], [2,2,2,2,2]], [3,3,3,3,3], 1)

        self.assertEqual(out, [10, 5])

    def test_weird_rotation(self):
        out = audio_wombat_test_helper1([[1,2,1,2,1,2]], [2,1,2,1,2,1], 2)[0]
        self.assertEqual(out, 6)

    def test_ignore_zeroes_in_filter(self):
        out = audio_wombat_test_helper1([[0,0,0,0,0,0,0,0]], [9,9,9,9,9,9,9,9], 1)[0]
        self.assertEqual(out, 0)

    def test_penalize_zeroes_in_image(self):
        out = audio_wombat_test_helper1([[9,9,9,9,9,9,9,9]], [0,0,0,0,0,0,0,0], 1)[0]
        self.assertEqual(out, 9*8)

    # def test_performance(self):
    #    frames = np.random.randint(5, size = (100, 129*982), dtype = np.int8)
    #    image = np.random.randint(5, size = (129*982), dtype = np.int8)

    #    start = perf_counter()
    #    out = audio_wombat_test_helper(frames, image, 129)[0]
    #    end = perf_counter()
    #    print(f"time: {end - start}")

    def test_performance1(self):
        frames = np.random.randint(5, size = (100000, 129*982), dtype = np.int8)
        image = np.random.randint(5, size = (129*982), dtype = np.int8)

        start = perf_counter()
        out = audio_wombat_test_helper1(frames, image, 129)[0]
        end = perf_counter()
        print(f"time: {end - start}")

def performance_data():
    for i in [1, 10, 100, 1_000]:
        frames = np.random.randint(5, size = (i, 129*982), dtype = np.int8)
        image = np.random.randint(5, size = (129*982), dtype = np.int8)
        start = perf_counter()
        out = audio_wombat_test_helper(frames, image, 129)[0]
        end = perf_counter()
        print(f"rust {i} time: {end - start}")
        start = perf_counter()
        out = audio_wombat_test_helper1(frames, image, 129)[0]
        end = perf_counter()
        print(f"cuda {i} time: {end - start}")
    for i in [10_000, 100_000]:
        frames = np.random.randint(5, size = (i, 129*982), dtype = np.int8)
        image = np.random.randint(5, size = (129*982), dtype = np.int8)
        start = perf_counter()
        out = audio_wombat_test_helper1(frames, image, 129)[0]
        end = perf_counter()
        print(f"cuda {i} time: {end - start}")




if __name__ == '__main__':
    unittest.main()

    #performance_data()


    #a = np.array([-1, 2, 4, 0, 5, 3, 6, 2, 1], dtype = np.float32)
    #b = np.array([3, 0, 2, 3, 4, 5, 4, 7, 2], dtype = np.float32)
    #c = np.zeros(9, dtype = np.float32)

    ##apoint, read_only_flag = a.__array_interface__["data"]
    ##bpoint, read_only_flag = b.__array_interface__["data"]
    ##cpoint, read_only_flag = c.__array_interface__["data"]
    #apoint = a.ctypes.data
    #bpoint = b.ctypes.data
    #cpoint = c.ctypes.data

    #print("here")

    #cu_maxmul(apoint, bpoint, cpoint, 3)
    #print(c)
