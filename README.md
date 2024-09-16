# Evaluation of Machine Learning Methods for Fire Risk Assessment from Satellite Imagery

This project exploits advanced geoscientific technologies and machine learning techniques to improve fire risk prediction and management. The primary objective is to develop a Convolutional Neural Network (CNN) that maps remotely sensed images to fire risk levels using a refined subset of the FireRisk dataset. The employed dataset contains 7,644 images categorised into five fire risk classes.

Additionally, a benchmark was conducted to evaluate the performance of more sophisticated supervised learning models on the constructed dataset. Specifically, leveraging transfer learning, the InceptionResNetV2 and ViT-B/16 models were fine-tuned.

## Dataset Development

Based on the FireRisk dataset, the Prometeus dataset has been designed to assess fire risk from remote sensing images. It comprises 7,644 labelled images; each categorised into one of five distinct fire risk classes (high, moderate, low non-burnable and water). Furthermore, the dataset was divided into training and validation sets in a ratio of 75:25.

![FireRisk overview image](https://github.com/joaocarlos/firerassess-cv_23_23/blob/main/images/prometheus_samples_resize.png)

| _This figure shows sample images of all 7 labels in our FireRisk dataset. The images measure 270 × 270 pixels, with a total of 91,872 image._

## Main contributions

-   A refined subset of the FireRisk dataset was used to train and evaluate the models.
-   A Convolutional Neural Network was developed to predict fire risk levels from satellite images.
-   The InceptionResNetV2 and ViT-B/16 models were fine-tuned to predict fire risk levels.
-   The performance of the CNN model was compared to the fine-tuned InceptionResNetV2 and ViT-B/16 models.

## Custom CNN Model

The CNN architecture begins with an input layer designed to handle images of 310 x 310 in size. The first convolutional layer employs 32 filters with a kernel size of 7 x 7. The architecture employs an increasing number of filters in subsequent convolutional layers, moving from 32 to 64 and then to 128.

![Custom CNN](https://github.com/joaocarlos/firerassess-cv_23_23/blob/main/images/MyCNN.pdf)

| _This figure describes the CNN model developed for the fire risk assessment._

## Benchmark Results

-   The InceptionResNetV2 was pre-trained on the ImageNet dataset and fine-tuned on the Prometeus dataset.
-   The ViT-B/16 model was pre-trained on the UnlabelledNAIP dataset and fine-tuned on the Prometeus dataset.
-   Using 1 GPU and batch size of 16 for training, and taking the results of the highest accuracy out of 80 epoches.

| Model             | Accuracy | Precision | Recall | F1-score |
| ----------------- | -------- | --------- | ------ | -------- |
| Custom CNN        | 0.72     | 0.72      | 0.72   | 0.72     |
| InceptionResNetV2 | 0.66     | 0.66      | 0.66   | 0.66     |
| ViT-B/16          | 0.70     | 0.71      | 0.70   | 0.70     |

_From the table we can draw the following conclusions:_

The custom CNN model achieved an accuracy and F1-score of 72% each, which is competitive with ViT-B/16 and outperforms InceptionResnetV2, which achieved 66% on both metrics. This result indicates that while the custom model performs robustly, there is room for improvement to reach the benchmark set by more mature architectures.

## Downloads

### Paper

[Evaluation of Machine Learning Methods for Fire Risk Assessment from Satellite Imagery](Evaluation of Machine Learning Methods for Fire Risk Assessment from Satellite Imagery)

### Dataset

[Prometeus dataset](https://drive.google.com/file/d/1vqx3fJnMXz4yMaa6h2DO-5HOySWQk1mN/view?usp=sharing)

Image naming convention: $(pointid)\_(grid_{code})\_(x_{coord})\_(y_{coord}).png$ (from the [FireRisk dataset](https://github.com/CharmonyShen/FireRisk/))

| Name       | Data Type            | Meaning                                                                                                                            |
| ---------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| FID        | integer              | ID of the data point in the file                                                                                                   |
| pointid    | integer              | unique ID of the data point in the WHP dataset                                                                                     |
| grid_code  | integer(from 1 to 5) | code for fire risk level                                                                                                           |
| class_desc | string (five values) | description of the fire risk level, corresponding to the grid_code, which are 1:Low, 2:Moderate, 3:High, 4:Non-burable and 5:water |
| x_coord    | number               | longitude coordinates of the grid centroid                                                                                         |
| y_coord    | number               | latitude coordinates of the grid centroid                                                                                          |

#### Pre-trained Checkpoints

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Custom</th>
<th valign="bottom">VIT</th>
<th valign="bottom">Inception</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/1DngVc7g86NH3Hxe4NkOhCK955r3pDju9/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1fmjdRcgkxZaUaGHQq1VZUCDkJ_kqXsG1/view?usp=sharing">download</a></td>
<td align="center"><a href=https://drive.google.com/file/d/1CikAEQSyi7S0Bd20jOYA2_UivPX6V3K2/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

### Citation

If you want to cite this work, please use the following BibTeX entry:

<!-- ```BibTeX
@inproceedings{bittencourt2023fireassessment,
      title={FireRisk: A Remote Sensing Dataset for Fire Risk Assessment with Benchmarks Using Supervised and Self-supervised Learning},
      author={Shuchang Shen and Sachith Seneviratne and Xinye Wanyan and Michael Kirley},
      year={2023},
      eprint={2303.07035},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` -->
