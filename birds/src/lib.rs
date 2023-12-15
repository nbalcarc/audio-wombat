use pyo3::prelude::*;
use std::slice;



extern {
    /// Calculates the output of the given neural network arrays, on CUDA
    fn audio_wombat_cuda(
        frames_pointer: *const u8,
        image_pointer: *const u8,
        output_pointer: *mut u32,

        image_size: u32,
        stride: u32,
        frame_count: u32,
    );

    fn maxmul(a: *const f32, b: *const f32, c: *mut f32, size: i32) -> ();


}





/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn cu_maxmul(a: u32, b: u32, c: u32, size: usize) -> PyResult<()> {
    unsafe { maxmul(a as *mut f32, b as *mut f32, c as *mut f32, size as i32) };
    Ok(())
}




fn audio_wombat_cuda_test(frame: &[u8], image: &[u8], image_size: usize, rotation_amount: usize) -> u32 {
    let mut min: u32 = std::u32::MAX;
    for i in 0..image_size / rotation_amount { //for each rotation 
        let mut accum: u32 = 0; //accumulator for each rotation
        let offset = i * rotation_amount;
        for t in 0..image_size { //represents one thread in cuda
            let thread_index = (offset + t) % image_size; //the index this thread will look at in the image
            if frame[t] == 0 { //if this index in the frame is 0, skip
                continue;
            }
            accum += (frame[t] as i32 - image[thread_index] as i32).abs() as u32;
        }
        if accum < min {
            min = accum;
        }
    }
    return min;
}


#[pyfunction]
/// Applies the frames to the images. Accepts numpy pointers.
fn audio_wombat(frames_pointer: usize, image_pointer: usize, output_pointer: usize, image_size: usize, rotation_amount: usize, frame_count: usize) {
    let frames: &[u8] = unsafe { slice::from_raw_parts(frames_pointer as *const u8, frame_count * image_size) };
    let image: &[u8] = unsafe { slice::from_raw_parts(image_pointer as *const u8, image_size) };
    let output_buffer: &mut [u32] = unsafe { slice::from_raw_parts_mut(output_pointer as *mut u32, frame_count) };

    for i in 0..frame_count { //for each frame
        println!("at frame {}", i);
        output_buffer[i] = audio_wombat_cuda_test(&frames[i*image_size..(i+1)*image_size], image, image_size, rotation_amount);
    }
}


#[pyfunction]
/// Applies the frames to the images. Accepts numpy pointers.
fn audio_wombat1(frames_pointer: usize, image_pointer: usize, output_pointer: usize, image_size: usize, stride: usize, frame_count: usize) {
    //let frames: &[u8] = unsafe { slice::from_raw_parts(frames_pointer as *const u8, frame_count * image_size) };
    //let image: &[u8] = unsafe { slice::from_raw_parts(image_pointer as *const u8, image_size) };
    let output_buffer: &mut [u32] = unsafe { slice::from_raw_parts_mut(output_pointer as *mut u32, frame_count) };
    for i in 0..output_buffer.len() {
        output_buffer[i] = std::u32::MAX;
    }

    unsafe { audio_wombat_cuda(
        frames_pointer as *const u8, 
        image_pointer as *const u8,
        output_pointer as *mut u32,
        image_size as u32,
        stride as u32, 
        frame_count as u32,
    )};
}


/// A Python module implemented in Rust.
#[pymodule]
fn birds(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(cu_maxmul, m)?)?;
    m.add_function(wrap_pyfunction!(audio_wombat, m)?)?;
    m.add_function(wrap_pyfunction!(audio_wombat1, m)?)?;
    Ok(())
}




