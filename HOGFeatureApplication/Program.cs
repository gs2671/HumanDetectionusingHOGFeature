using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.IO;


namespace HOGFeatureApplication
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("Starting the HOG processing....");

                #region Local Variables
                int i, j;              
                string[] posImages = Directory.GetFiles(@"../../Images/Training/Positive");//Get Positive Training Images
                string[] negImages = Directory.GetFiles(@"../../Images/Training/Negative");//Get Negative Training Images
                string[] posTestingImages = Directory.GetFiles(@"../../Images/Testing/Positive");//Get Positive Testing Images
                string[] negTestingImages = Directory.GetFiles(@"../../Images/Testing/Negative");//Get Negative Testing Images
                string positiveMeanVectorPath = @"../../Results/posmeanVector.csv";//Path to store Positive Mean Vector values
                string negativeMeanVectorPath = @"../../Results/negmeanVector.csv";//Path to store Negative Mean Vector values
                string positiveEuclideanDistPath = @"../../Results/poseucldist.csv";//Path to store Positive euclidean distance values
                string negativeEuclideanDistPath = @"../../Results/negeucldist.csv";//Path to store Negative euclidean distance values
                var csv_posmeanvect = new StringBuilder();                
                var csv_negmeanvect = new StringBuilder();
                var csv_poseucldist = new StringBuilder();
                var csv_negeucldist = new StringBuilder();

                double[,] pdscp = new double[10, 3780];//Store Positive Descriptor values
                double[] mpdscp = new double[3780];//Store Positive Mean Descriptor values
                double[] peucd = new double[10];//Store Positive Euclidean values

                double[,] ndscp = new double[10, 3780];//Store Negative Descriptor values
                double[] mndscp = new double[3780];//Store Negative Mean Descriptor values
                double[] neucd = new double[10];//Store Negative Euclidean values

                double[,] ptdscp = new double[5, 3780];//Store Positive Testing Images Descriptor values
                double[,] ntdscp = new double[5, 3780];//Store Negative Testing Images Descriptor values

                int picount,nicount;
                double psum, pavg, nsum, navg;
                double[] w = new double[3780];
                double eucdsum;
                double alpha = 0.01;
                var csv_imgDesc = new StringBuilder();
                #endregion

                #region For Positive Images
                Console.WriteLine("Calculating descriptors for positive images");
                //Getting feature vectors for positive images
                pdscp = GetFeatureVectorArray(posImages);

                //Calculating positive mean feature vector
                for (i = 0; i < 3780; i++)
                {
                    psum = 0.0;
                    pavg = 0.0;
                    for (j = 0; j < 10; j++)
                    {
                        psum = psum + pdscp[j, i];
                    }
                    pavg = psum / 10;
                    mpdscp[i] = pavg;
                    csv_posmeanvect.Append(mpdscp[i].ToString() + ",");
                    csv_posmeanvect.Append(Environment.NewLine);
                }
                File.WriteAllText(positiveMeanVectorPath, csv_posmeanvect.ToString());

                //Calculating Euclidean distance from positive mean vector to each image sample
                for (i = 0; i < 10; i++)
                {
                    eucdsum = 0.0;
                    for (j = 0; j < 3780; j++)
                    {
                        eucdsum = eucdsum + Math.Pow((mpdscp[i] - pdscp[i, j]), 2);
                    }
                    peucd[i] = Math.Sqrt(eucdsum);
                    csv_poseucldist.Append(peucd[i].ToString() + ",");
                    csv_poseucldist.Append(Environment.NewLine);
                }
                File.WriteAllText(positiveEuclideanDistPath, csv_poseucldist.ToString());
                #endregion

                #region For Negative Images
                Console.WriteLine("Calculating descriptors for negative images");
                //Getting feature vectors for negative images
                ndscp = GetFeatureVectorArray(negImages);

                //Calculating negative mean feature vector
                for (i = 0; i < 3780; i++)
                {
                    nsum = 0.0;
                    navg = 0.0;
                    for (j = 0; j < 10; j++)
                    {
                        nsum = nsum + ndscp[j, i];
                    }
                    navg = nsum / 10;
                    mndscp[i] = navg;
                    csv_negmeanvect.Append(mndscp[i].ToString() + ",");
                    csv_negmeanvect.Append(Environment.NewLine);                   
                }
                File.WriteAllText(negativeMeanVectorPath, csv_negmeanvect.ToString());

                //Calculating Euclidean distance from negative mean vector to each image sample
                for (i = 0; i < 10; i++)
                {
                    eucdsum = 0.0;
                    for (j = 0; j < 3780; j++)
                    {
                        eucdsum = eucdsum + Math.Pow((mndscp[i] - ndscp[i, j]), 2);
                    }
                    neucd[i] = Math.Sqrt(eucdsum);
                    csv_negeucldist.Append(neucd[i].ToString() + ",");
                    csv_negeucldist.Append(Environment.NewLine);
                }
                File.WriteAllText(negativeEuclideanDistPath, csv_negeucldist.ToString());
                #endregion

                #region Training a classifier    
                Console.WriteLine("Training a classifier");
                for (i = 0; i < 3780; i++)
                    w[i] = 1;
                picount = 0;
                nicount = 0;
                while (picount!= pdscp.GetLength(0) || nicount != ndscp.GetLength(0))
                {
                    psum = 0;
                    nsum = 0;
                    do
                    {
                        for (i = 0; i < 3780; i++)
                            psum = psum + (w[i] + pdscp[picount, i]);
                        if (psum < 0)
                        {
                            for (i = 0; i < 3780; i++)
                                w[i] = w[i] + (alpha * pdscp[picount, i]);
                        }
                    } while(psum < 0);
                    picount++;

                    do
                    {
                        for (i = 0; i < 3780; i++)
                            nsum = nsum + (w[i] + ndscp[nicount, i]);
                        if (nsum > 0)
                        {
                            for (i = 0; i < 3780; i++)
                                w[i] = w[i]- (alpha * ndscp[nicount, i]);
                        }
                    } while(nsum > 0);
                    nicount++;
                }
                #endregion

                #region Tesing the classifier for positive Images
                Console.WriteLine("Testing the classifier for positive Images");
                ptdscp = GetFeatureVectorArray(posTestingImages);
                for (i = 0; i < 5; i++)
                {
                    psum = 0;
                    for (j = 0; j < 3780; j++)
                        psum = psum + (w[j] + ptdscp[i, j]);
                    if (psum > 0)
                        Console.WriteLine("Image DOES HAVE human");
                    else
                        Console.WriteLine("Image DOES NOT HAVE human");
                }

                //Tesing the classifier for negative Images
                Console.WriteLine("Tesing the classifier for negative Images");
                ntdscp = GetFeatureVectorArray(negTestingImages);
                for (i = 0; i < 5; i++)
                {
                    nsum = 0;
                    for (j = 0; j < 3780; j++)
                        nsum = nsum + (w[j] + ntdscp[i, j]);
                    if (nsum < 0)
                        Console.WriteLine("Image DOES NOT HAVE human");
                    else
                        Console.WriteLine("Image DOES HAVE human");
                }
                #endregion

                Console.WriteLine("HOG Completed");
                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error:" + ex.Message);
                Console.ReadLine();
            }
        }

        #region Function to return intensity values as int[,]  array
        public static int[,] GetIntensity(Bitmap image)
        {
            int x, y;
            int[,] f = new int[image.Height, image.Width];
            for (y = 0; y <= image.Height - 1; y++)
            {
                for (x = 0; x <= image.Width - 1; x++)
                {
                    Color pixelColor = image.GetPixel(x, y);
                    f[y, x] = Convert.ToInt16((0.21 * pixelColor.R) + (0.72 * pixelColor.G) + (0.07 * pixelColor.B));//;F = 0.21 R + 0.72 G + 0.07 B
                }
            }
            return f;
        }
        #endregion

        #region Function to return gradient angle values
        public static int[,] GetGradAngle(int[,] f, int imgWidth, int imgHeight)
        {
            double gx, gy, gAng;
            int i, j;
            int[,] ga = new int[imgHeight, imgWidth];
            for (i = 0; i <= imgHeight - 1; i++)
                for (j = 0; j <= imgWidth - 1; j++)
                    ga[i, j] = 0;
            //Calculating Gradient Gradient Angle
            //Console.WriteLine("Calculating  Gradient Angle");
            for (i = 1; i <= imgHeight - 2; i++)
            {
                for (j = 1; j <= imgWidth - 2; j++)
                {
                    gx = f[i, j + 1] - f[i, j - 1];
                    gy = f[i + 1, j] - f[i - 1, j];
                    double temp;
                    temp = gy / gx;
                    if (double.IsNaN(temp))
                        gAng = 0;
                    else if (double.IsPositiveInfinity(temp))
                        gAng = 90;
                    else if (double.IsNegativeInfinity(temp))
                        gAng = 270;
                    else
                        gAng = -((Math.Atan(gy / gx) * 180) / Math.PI);//The negative sign is to convert the angle to X-Y coordinate system.
                    if (gAng < 0)
                        gAng = 360 + gAng;
                    ga[i, j] = (int)Math.Round(gAng, MidpointRounding.AwayFromZero);
                }
            }
            return ga;
        }
        #endregion

        #region Function to return gradient magnitude values
        public static int[,] GetGradMag(int[,] f, int imgWidth, int imgHeight)
        {
            double gx, gy, gMag;
            int i, j;
            int[,] gm = new int[imgHeight, imgWidth];
            for (i = 0; i <= imgHeight - 1; i++)
                for (j = 0; j <= imgWidth - 1; j++)
                    gm[i, j] = 0;
            //Calculating Gradient Magnitude          
            //Console.WriteLine("Calculating Gradient Magnitude");
            for (i = 1; i <= imgHeight - 2; i++)
            {
                for (j = 1; j <= imgWidth - 2; j++)
                {
                    gx = f[i, j + 1] - f[i, j - 1];
                    gy = f[i + 1, j] - f[i - 1, j];
                    gMag = Math.Sqrt((gx * gx) + (gy * gy)) / Math.Sqrt(2);
                    gm[i, j] = (int)Math.Round(gMag, MidpointRounding.AwayFromZero);
                }
            }
            return gm;
        }
        #endregion

        #region Function to return quantized gradient angle values
        public static int[,] GetQuantizedGradAngle(int[,] ga, int imgWidth, int imgHeight)
        {
            int[,] gq = new int[imgHeight, imgWidth];//Store gradient angle values.
            int i, j;
            for (i = 0; i <= imgHeight - 1; i++)
                for (j = 0; j <= imgWidth - 1; j++)
                    gq[i, j] = 0;
            //Calculating Quantized Gradient values
            //Console.WriteLine("Calculating quantized angle values....");
            for (i = 1; i <= imgHeight - 2; i++)
            {
                for (j = 1; j <= imgWidth - 2; j++)
                {
                    if ((ga[i, j] >= 0 && ga[i, j] < 20) || (ga[i, j] >= 180 && ga[i, j] < 200))
                        gq[i, j] = 1;
                    else if ((ga[i, j] >= 20 && ga[i, j] < 40) || (ga[i, j] >= 200 && ga[i, j] < 220))
                        gq[i, j] = 2;
                    else if ((ga[i, j] >= 40 && ga[i, j] < 60) || (ga[i, j] >= 220 && ga[i, j] < 240))
                        gq[i, j] = 3;
                    else if ((ga[i, j] >= 60 && ga[i, j] < 80) || (ga[i, j] >= 240 && ga[i, j] < 260))
                        gq[i, j] = 4;
                    else if ((ga[i, j] >= 80 && ga[i, j] < 100) || (ga[i, j] >= 260 && ga[i, j] < 280))
                        gq[i, j] = 5;
                    else if ((ga[i, j] >= 100 && ga[i, j] < 120) || (ga[i, j] >= 280 && ga[i, j] < 300))
                        gq[i, j] = 6;
                    else if ((ga[i, j] >= 120 && ga[i, j] < 140) || (ga[i, j] >= 300 && ga[i, j] < 320))
                        gq[i, j] = 7;
                    else if ((ga[i, j] >= 140 && ga[i, j] < 160) || (ga[i, j] >= 320 && ga[i, j] < 340))
                        gq[i, j] = 8;
                    else if ((ga[i, j] >= 160 && ga[i, j] < 180) || (ga[i, j] >= 340 && ga[i, j] < 360))
                        gq[i, j] = 9;
                    else if(ga[i, j] == 360)
                        gq[i, j] = 1;
                }
            }
            return gq;
        }
        #endregion

        #region Function to return histogram
        public static double[,] GetHistogram(int[,] gm, int[,] ga, int[,] gq, int imgWidth, int imgHeight)
        {
            double[,] hist = new double[129, 10];//Store histogram values 
            int k, l, i, j, sum = 0, num = 0, angleDiff, row = 1;
            double avg, wAvg;
            for (k = 1; k <= 128; k++)
                for (l = 1; l <= 9; l++)
                    hist[k, l] = 0;
            //Console.WriteLine("Calculating Histogram for each cell....");
            for (i = 16; i <= imgHeight - 24; i += 8)
            {
                //Console.WriteLine("i:" + i);
                for (j = 16; j <= imgWidth - 24; j += 8)
                {
                    //Console.WriteLine("j:" + j);
                    //Calculating the Average of each 8X8 cells 
                    //Console.WriteLine("Calculating average for cell " + row);
                    //Console.WriteLine("Row:" + row);
                    sum = 0;
                    num = 0;
                    for (k = i; k <= (i + 7); k++)
                    {
                        //if (k == 0 || k == imgHeight - 1)
                        //continue;
                        for (l = j; l <= (j + 7); l++)
                        {
                            if (l == 0 || l == imgWidth - 1)
                                continue;
                            sum = sum + gm[k, l];
                            num = num + 1;
                        }
                    }
                    avg = (double)sum / (double)num;

                    //Console.WriteLine("Calculating histogram for cell " + row);
                    //Diving each value in the 8X8 cell with the average and adding the value to the corresponding histogram array.                     
                    for (k = i; k <= (i + 7); k++)
                    {
                        //if (k == 0 || k == imgHeight - 1)
                        //continue;
                        for (l = j; l <= (j + 7); l++)
                        {
                            //Console.WriteLine("row:" + row);
                            angleDiff = 0;
                            //if (l == 0 || l == imgWidth - 1)//Ignoring the first and last row and column in the magnitude array
                            //  continue;
                            switch (gq[k, l])
                            {
                                case 1:
                                    angleDiff = (ga[k, l] >= 180) ? 10 - (ga[k, l] - 180) : 10 - ga[k, l];
                                    if (avg == 0)
                                        wAvg = 0;
                                    else
                                        wAvg = gm[k, l] / avg;
                                    //Distribute between current and previous bin if difference is greater then zero
                                    if (angleDiff >= 0)
                                    {
                                        hist[row, 1] = hist[row, 1] + (1 - (angleDiff / 20)) * wAvg;
                                        hist[row, 9] = hist[row, 9] + (angleDiff / 20) * wAvg;
                                    }
                                    else//Distribute between current and next bin if difference is less than zero
                                    {
                                        hist[row, 1] = hist[row, 1] + (1 - (Math.Abs(angleDiff) / 20)) * wAvg;
                                        hist[row, 2] = hist[row, 2] + (Math.Abs(angleDiff) / 20) * wAvg;
                                    }
                                    break;

                                case 2:
                                    angleDiff = (ga[k, l] >= 180) ? 30 - (ga[k, l] - 180) : 30 - ga[k, l];
                                    if (avg == 0)
                                        wAvg = 0;
                                    else
                                        wAvg = gm[k, l] / avg;
                                    if (angleDiff >= 0)
                                    {
                                        hist[row, 2] = hist[row, 2] + (1 - (angleDiff / 20)) * wAvg;
                                        hist[row, 1] = hist[row, 1] + (angleDiff / 20) * wAvg;
                                    }
                                    else
                                    {
                                        hist[row, 2] = hist[row, 2] + (1 - (Math.Abs(angleDiff) / 20)) * wAvg;
                                        hist[row, 3] = hist[row, 3] + (Math.Abs(angleDiff) / 20) * wAvg;
                                    }
                                    break;

                                case 3:
                                    angleDiff = (ga[k, l] >= 180) ? 50 - (ga[k, l] - 180) : 50 - ga[k, l];
                                    if (avg == 0)
                                        wAvg = 0;
                                    else
                                        wAvg = gm[k, l] / avg;
                                    if (angleDiff >= 0)
                                    {
                                        hist[row, 3] = hist[row, 3] + (1 - (angleDiff / 20)) * wAvg;
                                        hist[row, 2] = hist[row, 2] + (angleDiff / 20) * wAvg;
                                    }
                                    else
                                    {
                                        hist[row, 3] = hist[row, 3] + (1 - (Math.Abs(angleDiff) / 20)) * wAvg;
                                        hist[row, 4] = hist[row, 4] + (Math.Abs(angleDiff) / 20) * wAvg;
                                    }
                                    break;

                                case 4:
                                    angleDiff = (ga[k, l] >= 180) ? 70 - (ga[k, l] - 180) : 70 - ga[k, l];
                                    if (avg == 0)
                                        wAvg = 0;
                                    else
                                        wAvg = gm[k, l] / avg;
                                    if (angleDiff >= 0)
                                    {
                                        hist[row, 4] = hist[row, 4] + (1 - (angleDiff / 20)) * wAvg;
                                        hist[row, 3] = hist[row, 3] + (angleDiff / 20) * wAvg;
                                    }
                                    else
                                    {
                                        hist[row, 4] = hist[row, 4] + (1 - (Math.Abs(angleDiff) / 20)) * wAvg;
                                        hist[row, 5] = hist[row, 5] + (Math.Abs(angleDiff) / 20) * wAvg;
                                    }
                                    break;

                                case 5:
                                    angleDiff = (ga[k, l] >= 180) ? 90 - (ga[k, l] - 180) : 90 - ga[k, l];
                                    if (avg == 0)
                                        wAvg = 0;
                                    else
                                        wAvg = gm[k, l] / avg;
                                    if (angleDiff >= 0)
                                    {
                                        hist[row, 5] = hist[row, 5] + (1 - (angleDiff / 20)) * wAvg;
                                        hist[row, 4] = hist[row, 4] + (angleDiff / 20) * wAvg;
                                    }
                                    else
                                    {
                                        hist[row, 5] = hist[row, 5] + (1 - (Math.Abs(angleDiff) / 20)) * wAvg;
                                        hist[row, 6] = hist[row, 6] + (Math.Abs(angleDiff) / 20) * wAvg;
                                    }
                                    break;

                                case 6:
                                    angleDiff = (ga[k, l] >= 180) ? 110 - (ga[k, l] - 180) : 110 - ga[k, l];
                                    if (avg == 0)
                                        wAvg = 0;
                                    else
                                        wAvg = gm[k, l] / avg;
                                    if (angleDiff >= 0)
                                    {
                                        hist[row, 6] = hist[row, 6] + (1 - (angleDiff / 20)) * wAvg;
                                        hist[row, 5] = hist[row, 5] + (angleDiff / 20) * wAvg;
                                    }
                                    else
                                    {
                                        hist[row, 6] = hist[row, 6] + (1 - (Math.Abs(angleDiff) / 20)) * wAvg;
                                        hist[row, 7] = hist[row, 7] + (Math.Abs(angleDiff) / 20) * wAvg;
                                    }
                                    break;

                                case 7:
                                    angleDiff = (ga[k, l] >= 180) ? 130 - (ga[k, l] - 180) : 110 - ga[k, l];
                                    if (avg == 0)
                                        wAvg = 0;
                                    else
                                        wAvg = gm[k, l] / avg;
                                    if (angleDiff >= 0)
                                    {
                                        hist[row, 7] = hist[row, 7] + (1 - (angleDiff / 20)) * wAvg;
                                        hist[row, 6] = hist[row, 6] + (angleDiff / 20) * wAvg;
                                    }
                                    else
                                    {
                                        hist[row, 7] = hist[row, 7] + (1 - (Math.Abs(angleDiff) / 20)) * wAvg;
                                        hist[row, 8] = hist[row, 8] + (Math.Abs(angleDiff) / 20) * wAvg;
                                    }
                                    break;

                                case 8:
                                    angleDiff = (ga[k, l] >= 180) ? 150 - (ga[k, l] - 180) : 110 - ga[k, l];
                                    if (avg == 0)
                                        wAvg = 0;
                                    else
                                        wAvg = gm[k, l] / avg;
                                    if (angleDiff >= 0)
                                    {
                                        hist[row, 8] = hist[row, 8] + (1 - (angleDiff / 20)) * wAvg;
                                        hist[row, 7] = hist[row, 7] + (angleDiff / 20) * wAvg;
                                    }
                                    else
                                    {
                                        hist[row, 8] = hist[row, 8] + (1 - (Math.Abs(angleDiff) / 20)) * wAvg;
                                        hist[row, 9] = hist[row, 9] + (Math.Abs(angleDiff) / 20) * wAvg;
                                    }
                                    break;

                                case 9:
                                    angleDiff = (ga[k, l] >= 180) ? 170 - (ga[k, l] - 180) : 110 - ga[k, l];
                                    if (avg == 0)
                                        wAvg = 0;
                                    else
                                        wAvg = gm[k, l] / avg;
                                    if (angleDiff >= 0)
                                    {
                                        hist[row, 9] = hist[row, 9] + (1 - (angleDiff / 20)) * wAvg;
                                        hist[row, 8] = hist[row, 8] + (angleDiff / 20) * wAvg;
                                    }
                                    else
                                    {
                                        hist[row, 9] = hist[row, 9] + (1 - (Math.Abs(angleDiff) / 20)) * wAvg;
                                        hist[row, 1] = hist[row, 1] + (Math.Abs(angleDiff) / 20) * wAvg;
                                    }
                                    break;

                                default:
                                    Console.WriteLine("Default Value detected in switch lop"); break;
                            }
                        }
                    }
                    row++;
                }
            }
            return hist;
        }
        #endregion

        #region Function to return feature vector
        public static double[] GetFeatureVector(double[,] hist, int windowWidth, int windowHeight)
        {
            //Normalizing Histograms over blocks
            int cellcount = 0;
            int blkCount = 0;
            int ftCount = 0;
            int i = 1, j = 0, k, m;
            int noofblocks = ((windowHeight / 8) - 1) * ((windowWidth / 8) - 1);
            int cellsPerRow = windowWidth / 8;
            int cellsPerCol = windowHeight / 8;
            double blkAvg;
            double blkSum;
            double[] blkHist = new double[36];
            double[] dscp = new double[noofblocks * 36];

            while (blkCount != (noofblocks))
            {
                blkAvg = 0.0;
                blkSum = 0.0;
                j = 0;
                if ((i % cellsPerRow) == 0)
                    i++;
                for (k = 1; k <= 9; k++)
                {
                    blkHist[j] = hist[i, k];
                    j++;
                }
                for (k = 1; k <= 9; k++)
                {
                    blkHist[j] = hist[i + 1, k];
                    j++;
                }
                for (k = 1; k <= 9; k++)
                {
                    blkHist[j] = hist[i + cellsPerRow, k];
                    j++;
                }
                for (k = 1; k <= 9; k++)
                {
                    blkHist[j] = hist[i + cellsPerRow + 1, k];
                    j++;
                }

                for (m = 0; m < 36; m++)
                {
                    blkSum = blkSum + (blkHist[m] * blkHist[m]);
                }
                blkAvg = Math.Sqrt(blkSum);
                for (m = 0; m < 36; m++)
                {
                    if (blkAvg == 0)
                        blkHist[m] = 0;
                    else
                        blkHist[m] = blkHist[m] / blkAvg;
                    dscp[ftCount] = blkHist[m];
                    ftCount++;
                }

                i++;
                cellcount += 4;
                blkCount++;
            }
            return dscp;
        }
        #endregion

        #region Function to return feature vector array for images
        public static double[,] GetFeatureVectorArray(string[] images)
        {
            Bitmap testImage = null;
            int imgWidth, imgHeight, windowWidth, windowHeight, noofblocks, i,j, icount, k, l;

            //Initializing Arrays
            int[,] f;//Store original intensity values
            int[,] gm;//Store gradient magnitude values.
            int[,] ga;//Store gradient angle values.
            int[,] gq;//Store quantized angle values.
            double[,] hist;//Store histogram values              
            double[] d;
            double[,] dscp = new double[images.GetLength(0), 3780];//Store descriptor values
            icount = 0;
            string intensityPath = null;
            string gradMagPath = null;
            string gradAngPath = null;
            string gradQuanAngPath = null;
            string cellHistPath = null;
            string featVectorPath = null;
            var csv_intensity = new StringBuilder();
            var csv_magnitude = new StringBuilder();
            var csv_angle = new StringBuilder();
            var csv_quanangle = new StringBuilder();
            var csv_cellhist = new StringBuilder();
            var csv_featvector = new StringBuilder();

            foreach (string imagePath in images)
            {
                testImage = new Bitmap(imagePath);
                imgWidth = testImage.Width;
                imgHeight = testImage.Height;
                windowWidth = imgWidth - 32;
                windowHeight = imgHeight - 32;                
                noofblocks = ((windowHeight / 8) - 1) * ((windowWidth / 8) - 1);
                f = new int[imgHeight, imgWidth];//Store original intensity values
                gm = new int[imgHeight, imgWidth];//Store gradient magnitude values.
                ga = new int[imgHeight, imgWidth];//Store gradient angle values.
                gq = new int[imgHeight, imgWidth];//Store quantized angle values.
                hist = new double[241, 10];//Store histogram values              
                d = new double[noofblocks * 36];
               
                intensityPath = @"../../Results/intensity"+Path.GetFileNameWithoutExtension(imagePath)+".csv";
                gradMagPath = @"../../Results/gradMag" + Path.GetFileNameWithoutExtension(imagePath) + ".csv";
                gradAngPath = @"../../Results/gradAng" + Path.GetFileNameWithoutExtension(imagePath) + ".csv";
                gradQuanAngPath = @"../../Results/gradQuanAng" + Path.GetFileNameWithoutExtension(imagePath) + ".csv";
                cellHistPath = @"../../Results/cellHist" + Path.GetFileNameWithoutExtension(imagePath) + ".csv";
                featVectorPath = @"../../Results/featVector" + Path.GetFileNameWithoutExtension(imagePath) + ".csv";

                //Getting intensity in Grayscale in (r,c) format
                //Console.WriteLine("Getting the Intensity values");
                f = GetIntensity(testImage);

                //Calculating Gradient Magnitude and Gradient Angle                  
                gm = GetGradMag(f, imgWidth, imgHeight);
                ga = GetGradAngle(f, imgWidth, imgHeight);

                //Calculating Quantized Gradient values
                gq = GetQuantizedGradAngle(ga, imgWidth, imgHeight);

                //Calculating Histogram for each cell (8X8 pixels)
                hist = GetHistogram(gm, ga, gq, imgWidth, imgHeight);

                //Calculating Feature Vector
                d = GetFeatureVector(hist, windowWidth, windowHeight);
                for (i = 0; i < 3780; i++)
                    dscp[icount, i] = d[i];
                icount++;

                // Printing values into csv
                for (i = 0; i <= imgHeight - 1; i++)
                {
                    for (j = 0; j <= imgWidth - 1; j++)
                    {
                        csv_intensity.Append(f[i, j].ToString() + ",");
                        csv_magnitude.Append(gm[i, j].ToString() + ",");
                        csv_angle.Append(ga[i, j].ToString() + ",");
                        csv_quanangle.Append(gq[i, j].ToString() + ",");
                    }
                    csv_intensity.Append(Environment.NewLine);
                    csv_magnitude.Append(Environment.NewLine);
                    csv_angle.Append(Environment.NewLine);
                    csv_quanangle.Append(Environment.NewLine);
                }
                for (k = 1; k <= 128; k++)
                {
                    for (l = 1; l <= 9; l++)
                    {
                        csv_cellhist.Append(hist[k, l].ToString() + ",");
                    }
                    csv_cellhist.Append(Environment.NewLine);
                }

                for (i = 0; i < 3780; i++)
                {
                    csv_featvector.Append(d[i].ToString() + ",");
                    if(((i+1)%36)==0)
                        csv_featvector.Append(Environment.NewLine);
                }

                File.WriteAllText(intensityPath, csv_intensity.ToString());
                File.WriteAllText(gradMagPath, csv_magnitude.ToString());
                File.WriteAllText(gradAngPath, csv_angle.ToString());
                File.WriteAllText(gradQuanAngPath, csv_quanangle.ToString());
                File.WriteAllText(cellHistPath, csv_cellhist.ToString());
                File.WriteAllText(featVectorPath, csv_featvector.ToString());
            }
            return dscp;
        }
        #endregion
    }
}

