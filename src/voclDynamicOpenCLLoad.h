#ifndef __VOCL_DYNAMIC_OPENCL_LOAD_H__
#define __VOCL_DYNAMIC_OPENCL_LOAD_H__
#include <CL/opencl.h>

typedef cl_int
(*clGetPlatformIDsLocal)(cl_uint num_entries, cl_platform_id * platforms, cl_uint * num_platforms);

typedef cl_int
(*clGetDeviceIDsLocal)(cl_platform_id platform,
               cl_device_type device_type,
               cl_uint num_entries, cl_device_id * devices, cl_uint * num_devices);

typedef cl_context
(*clCreateContextLocal)(const cl_context_properties * properties,
                cl_uint num_devices,
                const cl_device_id * devices,
                void (CL_CALLBACK * pfn_notify) (const char *, const void *, size_t, void *),
                void *user_data, cl_int * errcode_ret);

typedef cl_command_queue
(*clCreateCommandQueueLocal)(cl_context context,
                     cl_device_id device,
                     cl_command_queue_properties properties, cl_int * errcode_ret);

typedef cl_program
(*clCreateProgramWithSourceLocal)(cl_context context,
                          cl_uint count,
                          const char **strings, const size_t * lengths, cl_int * errcode_ret);

typedef cl_int
(*clBuildProgramLocal)(cl_program program,
               cl_uint num_devices,
               const cl_device_id * device_list,
               const char *options,
               void (CL_CALLBACK * pfn_notify) (cl_program program, void *user_data),
               void *user_data);

typedef cl_kernel 
(*clCreateKernelLocal)(cl_program program, const char *kernel_name, cl_int * errcode_ret);

typedef cl_mem
(*clCreateBufferLocal)(cl_context context,
               cl_mem_flags flags, size_t size, void *host_ptr, cl_int * errcode_ret);

typedef cl_int
(*clEnqueueWriteBufferLocal)(cl_command_queue command_queue,
                     cl_mem buffer,
                     cl_bool blocking_write,
                     size_t offset,
                     size_t cb,
                     const void *ptr,
                     cl_uint num_events_in_wait_list,
                     const cl_event * event_wait_list, 
					 cl_event * event);

typedef cl_int
(*clSetKernelArgLocal)(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);

typedef cl_int
(*clEnqueueNDRangeKernelLocal)(cl_command_queue command_queue,
                       cl_kernel kernel,
                       cl_uint work_dim,
                       const size_t * global_work_offset,
                       const size_t * global_work_size,
                       const size_t * local_work_size,
                       cl_uint num_events_in_wait_list,
                       const cl_event * event_wait_list, cl_event * event);

/* Enqueued Commands for GPU memory read */
typedef cl_int
(*clEnqueueReadBufferLocal)(cl_command_queue command_queue,
                    cl_mem buffer,
                    cl_bool blocking_read,
                    size_t offset,
                    size_t cb,
                    void *ptr,
                    cl_uint num_events_in_wait_list,
                    const cl_event * event_wait_list, cl_event * event);

typedef cl_int (*clReleaseMemObjectLocal)(cl_mem memobj);
typedef cl_int (*clReleaseKernelLocal)(cl_kernel kernel);
typedef cl_int (*clFinishLocal)(cl_command_queue hInCmdQueue);

typedef cl_int
(*clGetContextInfoLocal)(cl_context context,
                 cl_context_info param_name,
                 size_t param_value_size, void *param_value, size_t * param_value_size_ret);

typedef cl_int
(*clGetProgramBuildInfoLocal)(cl_program program,
                      cl_device_id device,
                      cl_program_build_info param_name,
                      size_t param_value_size,
                      void *param_value, size_t * param_value_size_ret);

typedef cl_int
(*clGetProgramInfoLocal)(cl_program program,
                 cl_program_info param_name,
                 size_t param_value_size, void *param_value, size_t * param_value_size_ret);

typedef cl_int (*clReleaseProgramLocal)(cl_program program);
typedef cl_int (*clReleaseCommandQueueLocal)(cl_command_queue command_queue);
typedef cl_int (*clReleaseContextLocal)(cl_context context);

typedef cl_int
(*clGetDeviceInfoLocal)(cl_device_id device,
                cl_device_info param_name,
                size_t param_value_size, void *param_value, size_t * param_value_size_ret);

typedef cl_int
(*clGetPlatformInfoLocal)(cl_platform_id platform,
                  cl_platform_info param_name,
                  size_t param_value_size, void *param_value, size_t * param_value_size_ret);

typedef cl_int (*clFlushLocal)(cl_command_queue hInCmdQueue);

typedef cl_int (*clWaitForEventsLocal)(cl_uint num_events, const cl_event * event_list);

typedef cl_sampler
(*clCreateSamplerLocal)(cl_context context,
                cl_bool normalized_coords,
                cl_addressing_mode addressing_mode,
                cl_filter_mode filter_mode, cl_int * errcode_ret);

typedef cl_int
(*clGetCommandQueueInfoLocal)(cl_command_queue command_queue,
                      cl_command_queue_info param_name,
                      size_t param_value_size,
                      void *param_value, size_t * param_value_size_ret);

typedef void* (*clEnqueueMapBufferLocal)(cl_command_queue command_queue,
                         cl_mem buffer,
                         cl_bool blocking_map,
                         cl_map_flags map_flags,
                         size_t offset,
                         size_t cb,
                         cl_uint num_events_in_wait_list,
                         const cl_event * event_wait_list,
                         cl_event * event, cl_int * errcode_ret);

typedef cl_int (*clReleaseEventLocal)(cl_event event);
typedef cl_int
(*clGetEventProfilingInfoLocal)(cl_event event,
                        cl_profiling_info param_name,
                        size_t param_value_size,
                        void *param_value, size_t * param_value_size_ret);

typedef cl_int (*clReleaseSamplerLocal)(cl_sampler sampler);

typedef cl_int
(*clGetKernelWorkGroupInfoLocal)(cl_kernel kernel,
                         cl_device_id device,
                         cl_kernel_work_group_info param_name,
                         size_t param_value_size,
                         void *param_value, size_t * param_value_size_ret);
typedef cl_mem
(*clCreateImage2DLocal)(cl_context context,
                cl_mem_flags flags,
                const cl_image_format * image_format,
                size_t image_width,
                size_t image_height,
                size_t image_row_pitch, void *host_ptr, cl_int * errcode_ret);

typedef cl_int
(*clEnqueueCopyBufferLocal)(cl_command_queue command_queue,
                    cl_mem src_buffer,
                    cl_mem dst_buffer,
                    size_t src_offset,
                    size_t dst_offset,
                    size_t cb,
                    cl_uint num_events_in_wait_list,
                    const cl_event * event_wait_list, cl_event * event);
typedef cl_int (*clRetainEventLocal)(cl_event event);
typedef cl_int (*clRetainMemObjectLocal)(cl_mem memobj);
typedef cl_int (*clRetainKernelLocal)(cl_kernel kernel);
typedef cl_int (*clRetainCommandQueueLocal)(cl_command_queue command_queue);
typedef cl_int
(*clEnqueueUnmapMemObjectLocal)(cl_command_queue command_queue,
                        cl_mem memobj,
                        void *mapped_ptr,
                        cl_uint num_events_in_wait_list,
                        const cl_event * event_wait_list, cl_event * event);

#endif //__VOCL_DYNAMIC_OPENCL_LOAD_H__
