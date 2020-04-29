from pyueye import ueye
import numpy as np
import cv2
import sys
import ctypes
import time



class UEYECamera:

    def __init__(self, pathToParameterSets, fps):

        self.hCam = ueye.HIDS(0)            
        self.sInfo = ueye.SENSORINFO()
        self.cInfo = ueye.CAMINFO()
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.nBitsPerPixel = ueye.INT(24)            
        self.channels = 3                    
        self.m_nColorMode = ueye.INT()	
        self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
        self.timeStampsFilePath = "times.txt"
        self.timeStampsFile = open(self.timeStampsFilePath, 'w')
        self.FPS = ctypes.c_double(int(fps))
        self.pathToParameterSets = pathToParameterSets
    

        nRet = ueye.is_InitCamera(self.hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")


        # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
        nRet = ueye.is_GetCameraInfo(self.hCam, self.cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")

        # You can query additional information about the sensor type used in the camera
        nRet = ueye.is_GetSensorInfo(self.hCam, self.sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")


        if int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", self.m_nColorMode)
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()


        #loading parameter set file

        pParam = ueye.wchar_p()
        pParam.value = self.pathToParameterSets

        nRet = ueye.is_ParameterSet(self.hCam, ueye.IS_PARAMETERSET_CMD_LOAD_FILE, pParam, 0)
        if nRet != ueye.IS_SUCCESS:
            print("Setting parameter set error")

        #setting fps
        newFPS = ctypes.c_double(0)
        ueye.is_SetFrameRate(self.hCam, self.FPS, newFPS)
        if nRet != ueye.IS_SUCCESS:
            print("Setting fps ERROR")
        else:
            print("FPS is: ", newFPS)

        # Can be used to set the size and position of an "area of interest"(AOI) within an image
        nRet = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height

        # Prints out some information about the camera and the sensor
        print("Camera model:\t\t", self.sInfo.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", self.cInfo.SerNo.decode('utf-8'))
        print("Maximum image width:\t", self.width)
        print("Maximum image height:\t", self.height)
        print()

    def __del__(self):

        ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)

        # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(self.hCam)

        # Destroys the OpenCv windows
        self.timeStampsFile.close()

    def setMemroyForVideoCapture(self):
        nRet = ueye.is_AllocImageMem(self.hCam, self.width, self.height, self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)



        # Activates the camera's live video mode (free run mode)
        nRet = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, self.width, self.height, self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")
        else:
            print("Press q to leave the programm")


    def captureVideo(self):

        count = 0
        prevCaptureTime = 0
        imageInfo = ueye.UEYEIMAGEINFO()
        while(True):

            array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
            frame = np.reshape(array,(self.height.value, self.width.value, self.bytes_per_pixel))

            nRet = ueye.is_GetImageInfo(self.hCam, self.MemID, imageInfo , ueye.sizeof(imageInfo))
            if nRet != ueye.IS_SUCCESS:
                print("GET IMAGE INFO ERROR")

            captureTime = imageInfo.u64TimestampDevice
            if ((captureTime > prevCaptureTime) and (captureTime != 0)):

                exposureTime = ueye.double()
                retVal = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, exposureTime, 8)

                self.timeStampsFile.write(str(count).zfill(5) +" " + str(captureTime - 0) + " " + str(exposureTime) + "\n")
                cv2.imwrite("images/" + str(count).zfill(5) +".jpg", frame)

                count = count + 1
                prevCaptureTime = captureTime - 0
                cv2.imshow("captureVideo", frame)
            
       
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   
        
        cv2.destroyAllWindows()


    def captureCalibrationVideo(self):

        pParam = ueye.double()
        nRet = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, pParam, ueye.sizeof(pParam))
        if nRet != ueye.IS_SUCCESS:
            print("Error getting max exposure, error code: ", nRet)
        else:
            print("max exposure is: ",pParam.value)


        pParam = ueye.double()
        pParam.value = 0
        nRet = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, pParam, ueye.sizeof(pParam))


        count = 0
        prevCaptureTime = 0
        maxExposure = pParam.value
        currExposure = 0
        imageInfo = ueye.UEYEIMAGEINFO()
        while(currExposure <= maxExposure):

            array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
            frame = np.reshape(array,(self.height.value, self.width.value, self.bytes_per_pixel))

            nRet = ueye.is_GetImageInfo(self.hCam, self.MemID, imageInfo , ueye.sizeof(imageInfo))
            if nRet != ueye.IS_SUCCESS:
                print("GET IMAGE INFO ERROR")

            captureTime = imageInfo.u64TimestampDevice
            if ((captureTime > prevCaptureTime) and (captureTime != 0)):

                pParam = ueye.double()
                pParam.value = currExposure
                nRet = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, pParam, ueye.sizeof(pParam))

                currExposure += 0.01

                self.timeStampsFile.write(str(count).zfill(5) +" " + str(captureTime - 0) + " " + str(currExposure) + "\n")
                cv2.imwrite("images/" + str(count).zfill(5) +".jpg", frame)

                count = count + 1
                prevCaptureTime = captureTime - 0
                cv2.imshow("captureVideo", frame)
            
       
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   
        
        cv2.destroyAllWindows()
