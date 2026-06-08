program dist_matmul
   use mpi
   use hdf5
   implicit none
   integer :: rank, size, ierr
   integer :: N, M, p
   integer :: left_rank, right_rank, k, s, i, j, L
   integer :: req_r, req_s
   integer :: arg_status
   real(8), allocatable :: A_curr(:,:), A_next(:,:), B_loc(:,:), C_loc(:,:), C_read(:,:)
   real(8) :: tmp, diff, max_diff, rel_diff, max_rel_diff, max_expected_abs
   real(8) :: read_diff, max_read_diff, global_max_read_diff
   real(8) :: S1, S2, expected
   real(8) :: total_start, total_time, init_start, init_time
   real(8) :: compute_time, comm_time, io_start, io_time, validation_start, validation_time
   real(8) :: io_write_time, io_read_time, max_io_write_time, max_io_read_time
   real(8) :: t0, t1
   real(8) :: max_total_time, max_init_time, max_compute_time
   real(8) :: max_comm_time, max_io_time, max_validation_time
   real(8) :: global_max_diff, global_max_rel_diff, global_max_expected_abs
   integer(HID_T)   :: file_id, filespace, memspace, dset_id
   integer(HID_T)   :: fapl_id, dxpl_id
   integer(HSIZE_T) :: dims(2), count(2), offset(2)
   character(len=256) :: output_path

   call MPI_Init(ierr)
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)

   total_start = MPI_Wtime()
   init_start = total_start
   compute_time = 0.0d0
   comm_time = 0.0d0
   io_time = 0.0d0
   validation_time = 0.0d0

   call get_command_argument(1, output_path, status=arg_status)
   if (arg_status /= 0 .or. len_trim(output_path) == 0) then
      write(output_path, '("C_dist_", I0, "ranks.h5")') size
   end if

   ! TODO should be divisible by size
   N = 32768
   if (mod(N, size) /= 0) then
      if (rank == 0) print *, "N must be divisible by processes"
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
   end if
   M = N / size
   p = rank

   allocate(A_curr(N, M), A_next(N, M), B_loc(N, M), C_loc(N, M), C_read(N, M))

   ! Init arrays
   ! A_curr initially corresponds to columns p*M+1 to (p+1)*M
   do j = 1, M
      do i = 1, N
         A_curr(i, j) = real(i, 8) + real(p*M + j, 8)
         B_loc(i, j) = real(i, 8) * real(p*M + j, 8)
         C_loc(i, j) = 0.0d0
      end do
   end do

   left_rank = mod(rank - 1 + size, size)
   right_rank = mod(rank + 1, size)

   !$acc enter data copyin(A_curr, B_loc) create(C_loc, A_next)
   !$acc update device(C_loc)
   init_time = MPI_Wtime() - init_start

   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   if (rank == 0) then
      print *, "Starting grid computation..."
      print *, "Output file: ", trim(output_path)
   end if

   do s = 0, size - 1
      k = mod(p - s + size, size)

      ! Non-blocking communication
      t0 = MPI_Wtime()
      !$acc host_data use_device(A_curr, A_next)
      call MPI_Irecv(A_next, N*M, MPI_DOUBLE_PRECISION, left_rank, 0, MPI_COMM_WORLD, req_r, ierr)
      call MPI_Isend(A_curr, N*M, MPI_DOUBLE_PRECISION, right_rank, 0, MPI_COMM_WORLD, req_s, ierr)
      !$acc end host_data
      t1 = MPI_Wtime()
      comm_time = comm_time + (t1 - t0)

      ! Async computation
      t0 = MPI_Wtime()
      !$acc parallel loop collapse(2) present(C_loc, A_curr, B_loc) async(1)
      do j = 1, M
         do i = 1, N
            tmp = 0.0d0
            !$acc loop seq
            do L = 1, M
               tmp = tmp + A_curr(i, L) * B_loc(k*M + L, j)
            end do
            C_loc(i, j) = C_loc(i, j) + tmp
         end do
      end do

      ! Wait for both compute and comms
      !$acc wait(1)
      t1 = MPI_Wtime()
      compute_time = compute_time + (t1 - t0)

      t0 = MPI_Wtime()
      call MPI_Wait(req_r, MPI_STATUS_IGNORE, ierr)
      call MPI_Wait(req_s, MPI_STATUS_IGNORE, ierr)
      t1 = MPI_Wtime()
      comm_time = comm_time + (t1 - t0)

      if (s < size - 1) then
         ! Shift A_next -> A_curr
         t0 = MPI_Wtime()
         !$acc parallel loop collapse(2) present(A_curr, A_next) async(2)
         do j = 1, M
            do i = 1, N
               A_curr(i, j) = A_next(i, j)
            end do
         end do
         !$acc wait(2)
         t1 = MPI_Wtime()
         compute_time = compute_time + (t1 - t0)
      endif
   end do

   !$acc update self(C_loc)
   !$acc exit data delete(A_curr, B_loc, C_loc, A_next)

   ! CPU Check Correctness
   validation_start = MPI_Wtime()
   S1 = real(N,8) * real(N+1,8) / 2.0d0
   S2 = real(N,8) * real(N+1,8) * real(2*N+1,8) / 6.0d0
   max_diff = 0.0d0
   max_rel_diff = 0.0d0
   max_expected_abs = 0.0d0

   do j = 1, M
      do i = 1, N
         expected = real(p*M + j, 8) * (real(i,8) * S1 + S2)
         diff = abs(C_loc(i, j) - expected)
         rel_diff = diff / max(abs(expected), 1.0d0)
         if (diff > max_diff) max_diff = diff
         if (rel_diff > max_rel_diff) max_rel_diff = rel_diff
         if (abs(expected) > max_expected_abs) max_expected_abs = abs(expected)
      end do
   end do

   validation_time = MPI_Wtime() - validation_start
   call MPI_Reduce(max_diff, global_max_diff, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
   call MPI_Reduce(max_rel_diff, global_max_rel_diff, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
   call MPI_Reduce(max_expected_abs, global_max_expected_abs, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)

   if (rank == 0) then
      write(*,'("VALIDATION max_abs_error=", ES12.5, " max_rel_error=", ES12.5, &
      &" max_abs_expected=", ES12.5)') global_max_diff, global_max_rel_diff, global_max_expected_abs
   end if

   ! Parallel HDF5 Write
   io_start = MPI_Wtime()
   call h5open_f(ierr)
   call h5pcreate_f(H5P_FILE_ACCESS_F, fapl_id, ierr)
   call h5pset_fapl_mpio_f(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL, ierr)
   call h5fcreate_f(trim(output_path), H5F_ACC_TRUNC_F, file_id, ierr, access_prp=fapl_id)

   dims(1) = int(N, HSIZE_T)
   dims(2) = int(N, HSIZE_T)
   call h5screate_simple_f(2, dims, filespace, ierr)
   call h5dcreate_f(file_id, "C", H5T_NATIVE_DOUBLE, filespace, dset_id, ierr)
   call h5sclose_f(filespace, ierr)

   count(1) = int(N, HSIZE_T)
   count(2) = int(M, HSIZE_T)
   offset(1) = 0
   offset(2) = int(p * M, HSIZE_T)
   call h5dget_space_f(dset_id, filespace, ierr)
   call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, offset, count, ierr)

   call h5screate_simple_f(2, count, memspace, ierr)

   call h5pcreate_f(H5P_DATASET_XFER_F, dxpl_id, ierr)
   call h5pset_dxpl_mpio_f(dxpl_id, H5FD_MPIO_COLLECTIVE_F, ierr)

   call h5dwrite_f(dset_id, H5T_NATIVE_DOUBLE, C_loc, count, ierr, &
      file_space_id=filespace, mem_space_id=memspace, xfer_prp=dxpl_id)

   call h5pclose_f(dxpl_id, ierr)
   call h5sclose_f(memspace, ierr)
   call h5sclose_f(filespace, ierr)
   call h5dclose_f(dset_id, ierr)
   call h5pclose_f(fapl_id, ierr)
   call h5fclose_f(file_id, ierr)
   call h5close_f(ierr)

   io_write_time = MPI_Wtime() - io_start

   ! Parallel HDF5 readback of the same rank-local hyperslab.
   C_read = 0.0d0
   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   t0 = MPI_Wtime()
   call h5open_f(ierr)
   call h5pcreate_f(H5P_FILE_ACCESS_F, fapl_id, ierr)
   call h5pset_fapl_mpio_f(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL, ierr)
   call h5fopen_f(trim(output_path), H5F_ACC_RDONLY_F, file_id, ierr, access_prp=fapl_id)
   call h5dopen_f(file_id, "C", dset_id, ierr)

   call h5dget_space_f(dset_id, filespace, ierr)
   call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, offset, count, ierr)
   call h5screate_simple_f(2, count, memspace, ierr)

   call h5pcreate_f(H5P_DATASET_XFER_F, dxpl_id, ierr)
   call h5pset_dxpl_mpio_f(dxpl_id, H5FD_MPIO_COLLECTIVE_F, ierr)

   call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, C_read, count, ierr, &
      file_space_id=filespace, mem_space_id=memspace, xfer_prp=dxpl_id)

   call h5pclose_f(dxpl_id, ierr)
   call h5sclose_f(memspace, ierr)
   call h5sclose_f(filespace, ierr)
   call h5dclose_f(dset_id, ierr)
   call h5pclose_f(fapl_id, ierr)
   call h5fclose_f(file_id, ierr)
   call h5close_f(ierr)

   io_read_time = MPI_Wtime() - t0

   max_read_diff = 0.0d0
   do j = 1, M
      do i = 1, N
         read_diff = abs(C_read(i, j) - C_loc(i, j))
         if (read_diff > max_read_diff) max_read_diff = read_diff
      end do
   end do
   call MPI_Reduce(max_read_diff, global_max_read_diff, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)

   io_time = MPI_Wtime() - io_start
   total_time = MPI_Wtime() - total_start

   call MPI_Reduce(init_time, max_init_time, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
   call MPI_Reduce(compute_time, max_compute_time, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
   call MPI_Reduce(comm_time, max_comm_time, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
   call MPI_Reduce(io_time, max_io_time, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
   call MPI_Reduce(io_write_time, max_io_write_time, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
   call MPI_Reduce(io_read_time, max_io_read_time, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
   call MPI_Reduce(validation_time, max_validation_time, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
   call MPI_Reduce(total_time, max_total_time, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)

   if (rank == 0) then
      write(*,'("READBACK max_abs_error=", ES12.5)') global_max_read_diff
      ! write(*,'("TIMING_IO ranks=", I0, " parallel_write_s=", F12.6, &
      ! &" parallel_read_s=", F12.6)') size, max_io_write_time, max_io_read_time
      print *, "Finished parallel write/read to ", trim(output_path)
      write(*,'("TIMING ranks=", I0, " init_s=", F12.6, " computation_s=", F12.6, &
      &" communication_s=", F12.6, " io_s=", F12.6, " io_write_s=", F12.6, &
      &" io_read_s=", F12.6, " validation_s=", F12.6, &
      &" total_s=", F12.6)') size, max_init_time, max_compute_time, max_comm_time, &
         max_io_time, max_io_write_time, max_io_read_time, max_validation_time, max_total_time
   end if

   call MPI_Finalize(ierr)
end program dist_matmul
