"use client";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState, FC } from "react";
import { createPortal } from "react-dom";
import axios from "axios";
import { Button } from "./ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "./ui/form";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import Image from 'next/image';

const FaceImage = z
  .instanceof(File)
  .refine(
    (file) => ["image/jpeg", "image/png", "image/gif"].includes(file.type),
    {
      message: "Only JPEG, PNG, and GIF images are allowed.",
    }
  )
  .refine((file) => file.size <= 5 * 1024 * 1024, {
    message: "File size must be less than 5 MB.",
  });

const formSchema = z.object({
  Name: z.string(),
  PrisonerNumber: z.string(),
  Age: z.string(),
  Height: z.string(),
  Weight: z.string(),
  Gender: z.enum(["Male", "Female", "Others"]),
  FaceImages: z.array(FaceImage),
});

interface ModalProps {
  showDialog: boolean;
  currentImage: string | null;
  saveImage: () => void;
  cancelCapture: () => void;
}

const Modal: FC<ModalProps> = ({ showDialog, currentImage, saveImage, cancelCapture }) => {
  if (!showDialog) return null;

  return createPortal(
    <div className="fixed top-0 left-0 w-full h-full flex items-center justify-center bg-black bg-opacity-50 z-50">
      <div className="bg-white p-6 rounded-lg shadow-lg max-w-md w-full">
        <h2 className="text-xl font-bold mb-4">Captured Image</h2>
        {currentImage && <img src={currentImage} alt="Captured" className="rounded-lg mb-4 w-full" />}
        <div className="flex justify-between">
          <button onClick={saveImage} className="bg-green-600 text-white py-2 px-4 rounded">Save Image</button>
          <button onClick={cancelCapture} className="bg-red-600 text-white py-2 px-4 rounded">Cancel</button>
        </div>
      </div>
    </div>,
    document.body
  );
};

export function RegistrationForm() {
  const router = useRouter();
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      Name: "",
      PrisonerNumber: "",
      Age: "",
      Height: "",
      Weight: "",
      Gender: undefined,
      FaceImages: [],
    },
  });

  const [showCamera, setShowCamera] = useState(false);
  const [capturedImages, setCapturedImages] = useState<string[]>([]);
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [showSubmitButton, setShowSubmitButton] = useState(false);
  const [showDialog, setShowDialog] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const capturePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      if (context) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0);
        const image = canvas.toDataURL('image/png');
        setCapturedImages((prev) => [...prev, image]);
        setCurrentImage(image);
        setShowDialog(true);
      }
    }
  };
  
  const cancelCapture = () => {
    setCapturedImages([]);
    setShowCamera(false);
    setShowDialog(false); // Ensure the modal is closed
  };

  const cancelImageCapture = () => {
    setCapturedImages((prev) => prev.slice(0, -1)); // Remove the last captured image
    setCurrentImage(null);
    setShowDialog(false); // Close the dialog box
  };
  
  const saveImage = () => {
    // Convert data URLs to File objects and update form state
    const newFaceImages = capturedImages.map((dataUrl, index) => {
      if (dataUrl) {
        const arr = dataUrl.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while (n--) {
          u8arr[n] = bstr.charCodeAt(n);
        }
        return new File([u8arr], `captured_image_${index}.png`, { type: mime });
      }
      return null;
    }).filter(file => file !== null); // Filter out any null values
  
    form.setValue("FaceImages", newFaceImages);
    setShowDialog(false);
  };

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      alert("Submitting project");
      const formData = new FormData();
      formData.append("Name", values.Name);
      formData.append("PrisonerNumber", values.PrisonerNumber); // Include PrisonerNumber
      formData.append("Age", values.Age);
      formData.append("Height", values.Height);
      formData.append("Weight", values.Weight);
      formData.append("Gender", values.Gender);
  
      values.FaceImages.forEach((file) => {
        formData.append("FaceImages", file); // Use the key "FaceImages" for multiple files
      });
  
      console.log("FormData entries:");
      for (let pair of formData.entries()) {
        console.log(pair[0] + ': ' + pair[1]);
      }
  
      const res = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log("Response:", res.data);
  
      // Redirect or show success message
      alert("Registered Successfully");
      router.push('/'); // Example route
    } catch (error) {
      console.error("Failed to submit project", error);
    }
  };

  useEffect(() => {
    if (showCamera) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch(err => console.error(err));
    }

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, [showCamera]);

  return (
    <div className="px-32">
      <h1 className="text-4xl font-bold text-center mb-8">REGISTRATION</h1>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
          <div className="flex gap-4">
            <FormField
              control={form.control}
              name="Name"
              render={({ field }) => (
                <FormItem className="flex-1">
                  <FormLabel>Name</FormLabel>
                  <FormControl>
                    <Input placeholder="Name" type="text" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="PrisonerNumber"
              render={({ field }) => (
                <FormItem className="w-1/5">
                  <FormLabel>Prisoner Number</FormLabel>
                  <FormControl>
                    <Input placeholder="Prisoner Number" type="text" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>

          <span className="flex gap-8">
            <FormField
              control={form.control}
              name="Age"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Age</FormLabel>
                  <FormControl>
                    <Input type="text" placeholder="Age" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="Weight"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Weight</FormLabel>
                  <FormControl>
                    <Input type="text" placeholder="Weight" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="Gender"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Gender</FormLabel>
                  <FormControl>
                    <Select value={field.value} onValueChange={(value) => field.onChange(value)}>
                      <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="Gender" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Male">Male</SelectItem>
                        <SelectItem value="Female">Female</SelectItem>
                        <SelectItem value="Others">Others</SelectItem>
                      </SelectContent>
                    </Select>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="Height"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Height</FormLabel>
                  <FormControl>
                    <Input type="text" placeholder="Height (inch)" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </span>

          <Button type="button" onClick={() => { setShowCamera(true); setShowSubmitButton(true); }} className="bg-blue-500 text-white py-2 px-4 rounded">
            Capture Facial Data
          </Button>

          {/* Camera Section */}
          {showCamera && (
            <div className="flex flex-row items-start justify-center bg-white p-4 mt-8">
              <div className="flex flex-col items-center">
                <h1 className="text-4xl font-bold text-center mb-8">FACIAL DATA REGISTRATION</h1>
                <div>Please Capture according to the following Pose</div>
                <Image 
                  src={
                    capturedImages.length === 0 ? "/images/front.jpg" :
                    capturedImages.length === 1 ? "/images/left.jpg" :
                    capturedImages.length === 2 ? "/images/right.jpg" :"/images/ok.png"
                  }
                  height={1000}
                  width={1000}
                  alt=""
                  className="w-80 my-8 "
                />
                <video ref={videoRef} autoPlay className="border rounded-lg mb-4 w-full max-w-md" />
                <div className="flex flex-row gap-4 mb-2">
                  {capturedImages.length < 3 && (
                    <button type="button" onClick={capturePhoto} className="bg-blue-500 text-white py-2 px-4 rounded">
                      Capture Photo
                    </button>
                  )}
                  <button type="button" onClick={cancelCapture} className="bg-red-500 text-white py-2 px-4 rounded">
                    Cancel
                  </button>
                </div>
              </div>

              {/* Render captured images vertically */}
              <div className="flex flex-col gap-4 ml-4 h-full">
                {capturedImages.map((image, index) => (
                  <img key={index} src={image} alt={`Captured ${index + 1}`} className="rounded-lg w-full" style={{ height: '150px' }} />
                ))}
              </div>
            </div>
          )}

          {/* Final Submit Button */}
          <Button
            type="submit"
            className={`bg-green-500 text-white py-2 px-4 rounded mt-4 ml-4 ${!showSubmitButton ? 'opacity-50 cursor-not-allowed' : ''}`}
            // disabled={!showSubmitButton}
            onClick={() => onSubmit}
          >
            Final Submit
          </Button>
        </form>
      </Form>

      {/* Modal Component */}
      <Modal showDialog={showDialog} currentImage={currentImage} saveImage={saveImage} cancelCapture={cancelImageCapture} />
    </div>
  );
}