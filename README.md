# SSL-CPCD_IEEE_TMI
This is a repo on Self-supervised learning for endoscopy data

Contributors: Ziang Xu and Sharib Ali

## Paper: "SSL-CPCD: Self-supervised learning with composite pretext-class discrimination loss for improved generalisability in endoscopic image analaysis"

 ### Dataset details
 We explored three different datasets for our experiments on endoscopic image analysis as listed below:
 
 [LMIUC](https://zenodo.org/record/5827695#.ZD1Xsy_MIeY)
 
 [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
 
 [CVC-ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)
 
 [In-house unseen test data](https://doi.org/10.7303/syn52674005)

 A full training and test data list can also be downloaded [here](https://drive.google.com/drive/folders/1d0shq3lvgXfgxWqsJgbLjJ9_dXXbzXQA?usp=share_link). 

 [Checkpoint download for baseline vs our SSL-CPCD](https://drive.google.com/file/d/1vnjeoobxRUz7_eArGvk0awnuH_K41jS3/view?usp=share_link)
 
 ### Training, test and generalisation scripts 
 
 #### 1. Training for SSL-CPCD:
  <pre><code>
  cd ./SSL-CPCD
  python SSLCPCD_train.py
  </code></pre>
  ####  2. Training for downstream task:
  <pre><code>
  cd ./scriptDownstreamTaskTraining
  sh SSL_CPCD_finetune_UC_classification_train.sh
  sh SSL_CPCD_finetune_polyp_detection_train.sh
  sh SSL_CPCD_finetune_polyp_segmentation_train.sh
  </code></pre>
  ####  3. Test for each SSL-CPCD tasks
  <pre><code>
  cd ./test
  python test_classification.py
  python test_detection.py
  python test_segmentation.py
  </code></pre>