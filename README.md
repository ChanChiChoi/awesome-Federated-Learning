# awesome-Federated-Learning
The repository collects papers(mainly from arxiv.org), Frameworks, projects, datasets of federated learning on bellow themes:

> * [Papers][Introduction&Survey](https://github.com/ChanChiChoi/awesome-Federated-Learning#introduction--survey)  
> * [Papers&Statistical][Distributed Optimization](https://github.com/ChanChiChoi/awesome-Federated-Learning#distributed-optimization)
> * [Papers&Statistical][Non-IID and Model Personalization](https://github.com/ChanChiChoi/awesome-Federated-Learning#non-iid-and-model-personalization)
> * [Papers&Statistical][Semi-Supervised Learning](https://github.com/ChanChiChoi/awesome-Federated-Learning#semi-supervised-learning)
> * [Papers&Statistical][Vertical Federated Learning](https://github.com/ChanChiChoi/awesome-Federated-Learning#vertical-federated-learning)
> * [Papers&Statistical][Hierarchical Federated Learning && Horizontal Federated Learning](https://github.com/ChanChiChoi/awesome-Federated-Learning#hierarchical-federated-learning--horizontal-federated-learning)
> * [Papers&Statistical][Decentralized Federated Learning](https://github.com/ChanChiChoi/awesome-Federated-Learning#decentralized-federated-learning)
> * [Papers&Statistical][Federated Transfer Learning](https://github.com/ChanChiChoi/awesome-Federated-Learning#federated-transfer-learning)
> * [Papers&Statistical][Neural Architecture Search](https://github.com/ChanChiChoi/awesome-Federated-Learning#neural-architecture-search)
> * [Papers&Statistical][Continual Learning](https://github.com/ChanChiChoi/awesome-Federated-Learning#continual-learning)
> * [Papers&Statistical][Reinforcement Learning && Robotics](https://github.com/ChanChiChoi/awesome-Federated-Learning#reinforcement-learning--robotics)
> * [Papers&Statistical][Bayesian Learning](https://github.com/ChanChiChoi/awesome-Federated-Learning#bayesian-learning)
> * [Papers&Trustworthiness][Adversarial-Attack-and-Defense](https://github.com/ChanChiChoi/awesome-Federated-Learning#adversarial-attack-and-defense)
> * [Papers&Trustworthiness][Privacy](https://github.com/ChanChiChoi/awesome-Federated-Learning#privacy--homomorphic-encryption)
> * [Papers&Trustworthiness][Incentive Mechanism && Fairness](https://github.com/ChanChiChoi/awesome-Federated-Learning#incentive-mechanism--fairness)
> * [Papers&System][Communication-Efficiency](https://github.com/ChanChiChoi/awesome-Federated-Learning#computation-efficiency)
> * [Papers&System][Straggler Problem](https://github.com/ChanChiChoi/awesome-Federated-Learning#straggler-problem)
> * [Papers&System][Computation Efficiency](https://github.com/ChanChiChoi/awesome-Federated-Learning#computation-efficiency)
> * [Papers&System][Wireless Communication && Cloud Computing && Networking](https://github.com/ChanChiChoi/awesome-Federated-Learning#wireless-communication--cloud-computing--networking)
> * [Papers&System][System Design](https://github.com/ChanChiChoi/awesome-Federated-Learning#system-design)
> * [Papers&Models][Models](https://github.com/ChanChiChoi/awesome-Federated-Learning#models)
> * [Papers&Applications][Natural language Processing](https://github.com/ChanChiChoi/awesome-Federated-Learning#natural-language-processing)
> * [Papers&Applications][Computer Vision](https://github.com/ChanChiChoi/awesome-Federated-Learning#computer-vision)
> * [Papers&Applications][Health Care](https://github.com/ChanChiChoi/awesome-Federated-Learning#health-care)
> * [Papers&Applications][Transportation](https://github.com/ChanChiChoi/awesome-Federated-Learning#transportation)
> * [Papers&Applications][Recommendation System](https://github.com/ChanChiChoi/awesome-Federated-Learning#recommendation-system)
> * [Papers&Applications][Speech Recognition](https://github.com/ChanChiChoi/awesome-Federated-Learning#speech-recognition)  
> * [Papers&Applications][Finance && Blockchain](https://github.com/ChanChiChoi/awesome-Federated-Learning#finance--blockchain)
> * [Papers&Applications][Smart City && Other Applications](https://github.com/ChanChiChoi/awesome-Federated-Learning#smart-city--other-applications)
> * [Papers&Others][uncategorized](https://github.com/ChanChiChoi/awesome-Federated-Learning#uncategorized)
> * [Blogs&&Tutorials](https://github.com/ChanChiChoi/awesome-Federated-Learning#blogs--tutorials)
> * [Framework](https://github.com/ChanChiChoi/awesome-Federated-Learning#framework)  
> * [Projects](https://github.com/ChanChiChoi/awesome-Federated-Learning#projects)
> * [Datasets && Benchmark](https://github.com/ChanChiChoi/awesome-Federated-Learning#datasets--benchmark)
> * [Scholars](https://github.com/ChanChiChoi/awesome-Federated-Learning#scholars)
> * [Conferences and Workshops](https://github.com/ChanChiChoi/awesome-Federated-Learning#conferences-and-workshops)
> * [Company](https://github.com/ChanChiChoi/awesome-Federated-Learning#company)

also, some papers and links collected from:
- [1-] [chaoyanghe/Awesome-Federated-Learning](https://github.com/chaoyanghe/Awesome-Federated-Learning)
- [2] [weimingwill/awesome-federated-learning](https://github.com/weimingwill/awesome-federated-learning)
- [3] [lokinko/Federated-Learning](https://github.com/lokinko/Federated-Learning)
- [4-] [tushar-semwal/awesome-federated-computing](https://github.com/tushar-semwal/awesome-federated-computing)
- [5-] [poga/awesome-federated-learning](https://github.com/poga/awesome-federated-learning)
- [6-] [timmers/awesome-federated-learning](https://github.com/timmers/awesome-federated-learning)
- [7-] [innovation-cat/Awesome-Federated-Machine-Learning](https://github.com/innovation-cat/Awesome-Federated-Machine-Learning)
- [8-] [ZeroWangZY/federated-learning](https://github.com/ZeroWangZY/federated-learning)
- [9-] [lee-man/federated-learning](https://github.com/lee-man/federated-learning)
- [10-] [albarqouni/Federated-Learning-In-Healthcare](https://github.com/albarqouni/Federated-Learning-In-Healthcare)


ps:LM:Linear Models; DM:Decision Trees; NN:Neural Networks; CM:Cryptographic Methods; DP:Differential Privacy; MA:Model Aggregation

---
## Introduction && Survey
- Wagner I, Eckhoff D. [Technical privacy metrics: a systematic survey](https://arxiv.org/pdf/1512.00327)[J]. ACM Computing Surveys (CSUR), 2018, 51(3): 1-38.
- Ben-Nun T, Hoefler T. [Demystifying parallel and distributed deep learning: An in-depth concurrency analysis](https://arxiv.org/pdf/1802.09941)[J]. ACM Computing Surveys (CSUR), 2019, 52(4): 1-43.
- Vepakomma P, Swedish T, Raskar R, et al. [No Peek: A Survey of private distributed deep learning](https://arxiv.org/pdf/1812.03288.pdf)[J]. arXiv preprint arXiv:1812.03288, 2018.
- [TIST]Qiang Yang, Yang Liu, Tianjian Chen, Yongxin Tong .[Federated Machine Learning: Concept and Applications](https://arxiv.org/pdf/1902.04885) [J]. arXiv preprint arXiv:1902.04885.
- Han Y, Wang X, Leung V, et al. [Convergence of Edge Computing and Deep Learning: A Comprehensive Survey](https://arxiv.org/pdf/1907.08349.pdf)[J]. arXiv preprint arXiv:1907.08349, 2019.
- Qinbin Li, Zeyi Wen, Bingsheng He .[Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection](https://arxiv.org/pdf/1907.09693) [J]. arXiv preprint arXiv:1907.09693.
- Solmaz Niknam, Harpreet S. Dhillon, Jeffery H. Reed .[Federated Learning for Wireless Communications: Motivation, Opportunities and Challenges](https://arxiv.org/pdf/1908.06847) [J]. arXiv preprint arXiv:1908.06847.
- Tian Li, Anit Kumar Sahu, Ameet Talwalkar, Virginia Smith .[Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/pdf/1908.07873) [J]. arXiv preprint arXiv:1908.07873.
- Wei Yang Bryan Lim, Nguyen Cong Luong, Dinh Thai Hoang, Yutao Jiao, Ying-Chang Liang, Qiang Yang, Dusit Niyato, Chunyan Miao .[Federated Learning in Mobile Edge Networks: A Comprehensive Survey](https://arxiv.org/pdf/1909.11875) [J]. arXiv preprint arXiv:1909.11875.
- D. Verma, S. Calo, S. Witherspoon, E. Bertino, A. Abu Jabal, A. Swami, G. Cirincione, S. Julier, G. White, G. de Mel, G. Pearson .[Federated Learning for Coalition Operations](https://arxiv.org/pdf/1910.06799) [J]. arXiv preprint arXiv:1910.06799.
- Hsieh K. [Machine Learning Systems for Highly-Distributed and Rapidly-Growing Data](https://arxiv.org/pdf/1910.08663)[J]. arXiv preprint arXiv:1910.08663, 2019.
- Bhardwaj K, Suda N, [Marculescu R. EdgeAI: A Vision for Deep Learning in IoT Era](https://arxiv.org/pdf/1910.10356)[J]. IEEE Design & Test, 2019.
- Jie Xu, Fei Wang .[Federated Learning for Healthcare Informatics](https://arxiv.org/pdf/1911.06270) [J]. arXiv preprint arXiv:1911.06270.
- Lan Q, Zhang Z, Du Y, et al. [An Introduction to Communication Efficient Edge Machine Learning](https://arxiv.org/pdf/1912.01554)[J]. arXiv preprint arXiv:1912.01554, 2019.
- Anudit Nagar .[Privacy-Preserving Blockchain Based Federated Learning with Differential Data Sharing](https://arxiv.org/pdf/1912.04859) [J]. arXiv preprint arXiv:1912.04859.
- [good]Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Keith Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, Rafael G.L. D'Oliveira, Salim El Rouayheb, David Evans, Josh Gardner, Zachary Garrett, Adrià Gascón, Badih Ghazi, Phillip B. Gibbons, Marco Gruteser, Zaid Harchaoui, Chaoyang He, Lie He, Zhouyuan Huo, Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara Javidi, Gauri Joshi, Mikhail Khodak, Jakub Konečný, Aleksandra Korolova, Farinaz Koushanfar, Sanmi Koyejo, Tancrède Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock, Ayfer Özgür, Rasmus Pagh, Mariana Raykova, Hang Qi, Daniel Ramage, Ramesh Raskar, Dawn Song, Weikang Song, Sebastian U. Stich, Ziteng Sun, Ananda Theertha Suresh, Florian Tramèr, Praneeth Vepakomma, Jianyu Wang, Li Xiong, Zheng Xu, Qiang Yang, Felix X. Yu, Han Yu, Sen Zhao .[Advances and Open Problems in Federated Learning](https://arxiv.org/pdf/1912.04977) [J]. arXiv preprint arXiv:1912.04977.
- Shi Y, Yang K, Jiang T, et al. [Communication-efficient edge AI: Algorithms and systems](https://arxiv.org/pdf/2002.09668)[J]. arXiv preprint arXiv:2002.09668, 2020.
- Ahmed Imteaj, Urmish Thakker, Shiqiang Wang, Jian Li, M. Hadi Amini .[Federated Learning for Resource-Constrained IoT Devices: Panoramas and State-of-the-art](https://arxiv.org/pdf/2002.10610) [J]. arXiv preprint arXiv:2002.10610.
- Yilun Jin, Xiguang Wei, Yang Liu, Qiang Yang .[A Survey towards Federated Semi-supervised Learning](https://arxiv.org/pdf/2002.11545) [J]. arXiv preprint arXiv:2002.11545.
- Lingjuan Lyu, Han Yu, Qiang Yang .[Threats to Federated Learning: A Survey](https://arxiv.org/pdf/2003.02133) [J]. arXiv preprint arXiv:2003.02133.
- Viraj Kulkarni, Milind Kulkarni, Aniruddha Pant .[Survey of Personalization Techniques for Federated Learning](https://arxiv.org/pdf/2003.08673) [J]. arXiv preprint arXiv:2003.08673.
- Christopher Briggs, Zhong Fan, Peter Andras .[A Review of Privacy Preserving Federated Learning for Private IoT Analytics](https://arxiv.org/pdf/2004.11794) [J]. arXiv preprint arXiv:2004.11794.
- Yi Liu, Xingliang Yuan, Zehui Xiong, Jiawen Kang, Xiaofei Wang, Dusit Niyato .[Federated Learning for 6G Communications: Challenges, Methods, and Future Directions](https://arxiv.org/pdf/2006.02931) [J]. arXiv preprint arXiv:2006.02931.
- Seyyedali Hosseinalipour, Christopher G. Brinton, Vaneet Aggarwal, Huaiyu Dai, Mung Chiang .[From Federated Learning to Fog Learning: Towards Large-Scale Distributed Machine Learning in Heterogeneous Wireless Networks](https://arxiv.org/pdf/2006.03594) [J]. arXiv preprint arXiv:2006.03594.
- Gupta A, Lanteigne C, Kingsley S. [SECure: A Social and Environmental Certificate for AI Systems](https://arxiv.org/pdf/2006.06217)[J]. arXiv preprint arXiv:2006.06217, 2020.


## Distributed Optimization
- Konečný J, McMahan B, Ramage D. [Federated optimization: Distributed optimization beyond the datacenter](https://arxiv.org/pdf/1511.03575)[J]. arXiv preprint arXiv:1511.03575, 2015.
- [Baseline]Brendan McMahan H, Moore E, Ramage D, et al. [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)[J]. arXiv, 2016: arXiv: 1602.05629.
- Jakub Konečný, H. Brendan McMahan, Daniel Ramage, Peter Richtárik .[Federated Optimization: Distributed Machine Learning for On-Device  Intelligence](https://arxiv.org/pdf/1610.02527) [J]. arXiv preprint arXiv:1610.02527.
- [NIPS]Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, Ameet Talwalkar .[Federated Multi-Task Learning](https://arxiv.org/pdf/1705.10467) [J]. arXiv preprint arXiv:1705.10467.
- Jiang Z, Balu A, Hegde C, et al. [Collaborative Deep Learning in Fixed Topology Networks](https://arxiv.org/pdf/1706.07880.pdf)[J]. arXiv preprint arXiv:1706.07880, 2017.
- Jakub Konečný .[Stochastic, Distributed and Federated Optimization for Machine Learning](https://arxiv.org/pdf/1707.01155) [J]. arXiv preprint arXiv:1707.01155.
- Wang S, Tuor T, Salonidis T, et al. [Adaptive Federated Learning in Resource Constrained Edge Computing Systems](https://arxiv.org/pdf/1804.05271.pdf)[J]. arXiv preprint arXiv:1804.05271, 2018.
- Stich S U.[Local SGD converges fast and communicates little](https://arxiv.org/pdf/1805.09767.pdf)[J]. arXiv preprint arXiv:1805.09767, 2018.
- Tianyi Chen, Georgios B. Giannakis, Tao Sun, Wotao Yin[LAG: Lazily Aggregated Gradient for Communication-Efficient Distributed Learning](https://arxiv.org/abs/1805.09965) [J]. arXiv preprint arXiv:1805.09965.
- Agarwal, Naman, et al. [CpSGD: Communication-Efficient and Differentially-Private Distributed SGD.](https://arxiv.org/abs/1805.10559) NIPS’18 Proceedings of the 32nd International Conference on Neural Information Processing Systems, vol. 31, 2018, pp. 7575–7586.
- Lin T, Stich S U, Patel K K, et al. [Don't Use Large Mini-Batches, Use Local SGD](https://arxiv.org/pdf/1808.07217)[J]. arXiv preprint arXiv:1808.07217, 2018.
- Wang J, Joshi G. [Cooperative SGD: A unified framework for the design and analysis of communication-efficient SGD algorithms](https://arxiv.org/pdf/1808.07576)[J]. arXiv preprint arXiv:1808.07576, 2018.
- Koskela A, Honkela A. [Learning Rate Adaptation for Federated and Differentially Private Learning](https://arxiv.org/pdf/1809.03832)[J]. arXiv preprint arXiv:1809.03832, 2018.
- Bui T D, Nguyen C V, Swaroop S, et al. [Partitioned variational inference: A unified framework encompassing federated and continual learning](https://arxiv.org/pdf/1811.11206)[J]. arXiv preprint arXiv:1811.11206, 2018.
- Li T, Sahu A K, Zaheer M, et al. [Federated optimization in heterogeneous networks](https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf)[J]. Proceedings of Machine Learning and Systems, 2020, 2: 429-450.<br>[code:[litian96/FedProx](https://github.com/litian96/FedProx)]
- Anit Kumar Sahu, Tian Li, Maziar Sanjabi, Manzil Zaheer, Ameet Talwalkar, Virginia Smith .[On the Convergence of Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127) [J]. arXiv preprint arXiv:1812.06127.
- Mohri M, Sivek G, Suresh A T. [Agnostic federated learning](https://arxiv.org/pdf/1902.00146)[J]. arXiv preprint arXiv:1902.00146, 2019.
- Neel Guha, Ameet Talwalkar, Virginia Smith .[One-Shot Federated Learning](https://arxiv.org/pdf/1902.11175) [J]. arXiv preprint arXiv:1902.11175.
- Xie C, Koyejo S, Gupta I. [Asynchronous federated optimization](https://arxiv.org/pdf/1903.03934)[J]. arXiv preprint arXiv:1903.03934, 2019.
- Eichner H, Koren T, McMahan H B, et al. [Semi-cyclic stochastic gradient descent](https://arxiv.org/pdf/1904.10120)[J]. arXiv preprint arXiv:1904.10120, 2019.
- Thakkar O, Andrew G, McMahan H B. [Differentially private learning with adaptive clipping](https://arxiv.org/pdf/1905.03871)[J]. arXiv preprint arXiv:1905.03871, 2019.
- [ICML]Mikhail Yurochkin, Mayank Agarwal, Soumya Ghosh, Kristjan Greenewald, Trong Nghia Hoang, Yasaman Khazaeni .[Bayesian Nonparametric Federated Learning of Neural Networks](https://arxiv.org/pdf/1905.12022) [J]. arXiv preprint arXiv:1905.12022.<br>[code:[IBM/probabilistic-federated-neural-matching](https://github.com/IBM/probabilistic-federated-neural-matching)]
- [good]Luca Corinzia, Joachim M. Buhmann .[Variational Federated Multi-Task Learning](https://arxiv.org/pdf/1906.06268) [J]. arXiv preprint arXiv:1906.06268.
- Avishek Ghosh, Justin Hong, Dong Yin, Kannan Ramchandran .[Robust Federated Learning in a Heterogeneous Environment](https://arxiv.org/pdf/1906.06629) [J]. arXiv preprint arXiv:1906.06629.
- Ghazi B, Pagh R, Velingker A. [Scalable and differentially private distributed aggregation in the shuffled model](https://arxiv.org/pdf/1906.08320)[J]. arXiv preprint arXiv:1906.08320, 2019.
- Khaled A, Mishchenko K, Richtárik P. [First analysis of local gd on heterogeneous data](https://arxiv.org/pdf/1909.04715)[J]. arXiv preprint arXiv:1909.04715, 2019.
- Khaled A, Richtárik P. [Gradient descent with compressed iterates](https://arxiv.org/pdf/1909.04716)[J]. arXiv preprint arXiv:1909.04716, 2019.
- Khaled A, Mishchenko K, Richtárik P. [Tighter theory for local SGD on identical and heterogeneous data](https://arxiv.org/pdf/1909.04746.pdf)[C]//International Conference on Artificial Intelligence and Statistics. PMLR, 2020: 4519-4529.
- Li B, Cen S, Chen Y, et al. [Communication-efficient distributed optimization in networks with gradient tracking](https://arxiv.org/pdf/1909.05844)[J]. arXiv preprint arXiv:1909.05844, 2019.
- Wei Liu, Li Chen, Yunfei Chen, Wenyi Zhang .[Accelerating Federated Learning via Momentum Gradient Descent](https://arxiv.org/pdf/1910.03197) [J]. arXiv preprint arXiv:1910.03197.
- Chaoyang He, Conghui Tan, Hanlin Tang, Shuang Qiu, Ji Liu .[Central Server Free Federated Learning over Single-sided Trust Social Networks](https://arxiv.org/pdf/1910.04956) [J]. arXiv preprint arXiv:1910.04956.
- [ICML][no IID]Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, Ananda Theertha Suresh .[SCAFFOLD: Stochastic Controlled Averaging for On-Device Federated Learning](https://arxiv.org/pdf/1910.06378) [J]. arXiv preprint arXiv:1910.06378.<br>[video:[scaffold-stochastic-controlled-averaging-for-federated-learning](https://slideslive.com/38927610/scaffold-stochastic-controlled-averaging-for-federated-learning)]
- Xin Yao, Tianchi Huang, Rui-Xiao Zhang, Ruiyu Li, Lifeng Sun .[Federated Learning with Unbiased Gradient Aggregation and Controllable Meta Updating](https://arxiv.org/pdf/1910.08234) [J]. arXiv preprint arXiv:1910.08234.
- Farzin Haddadpour, Mehrdad Mahdavi .[On the Convergence of Local Descent Methods in Federated Learning](https://arxiv.org/pdf/1910.14425) [J]. arXiv preprint arXiv:1910.14425.
- Saeedeh Parsaeefard, Iman Tabrizian, Alberto Leon Garcia .[Representation of Federated Learning via Worst-Case Robust Optimization Theory](https://arxiv.org/pdf/1912.05571) [J]. arXiv preprint arXiv:1912.05571.
- Sharma P, Khanduri P, Bulusu S, et al. [Parallel Restarted SPIDER--Communication Efficient Distributed Nonconvex Optimization with Optimal Computation Complexity](https://arxiv.org/pdf/1912.06036)[J]. arXiv preprint arXiv:1912.06036, 2019.
- Jakovetić D, Bajović D, Xavier J, et al. [Primal–Dual Methods for Large-Scale and Distributed Convex Optimization and Data Analytics](https://arxiv.org/pdf/1912.08546.pdf)J]. Proceedings of the IEEE, 2020, 108(11): 1923-1938.
- Chraibi S, Khaled A, Kovalev D, et al. [Distributed Fixed Point Methods with Compressed Iterates](https://arxiv.org/pdf/1912.09925)[J]. arXiv preprint arXiv:1912.09925, 2019.
- Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith .[FedDANE: A Federated Newton-Type Method](https://arxiv.org/pdf/2001.01920) [J]. arXiv preprint arXiv:2001.01920.
- Zhouyuan Huo, Qian Yang, Bin Gu, Lawrence Carin. Heng Huang .[Faster On-Device Training Using New Federated Momentum Algorithm](https://arxiv.org/pdf/2002.02090) [J]. arXiv preprint arXiv:2002.02090.
- Filip Hanzely, Peter Richtárik .[Federated Learning of a Mixture of Global and Local Models](https://arxiv.org/pdf/2002.05516) [J]. arXiv preprint arXiv:2002.05516.
- [ICLR]Hongyi Wang, Mikhail Yurochkin, Yuekai Sun, Dimitris Papailiopoulos, Yasaman Khazaeni .[Federated Learning with Matched Averaging](https://arxiv.org/pdf/2002.06440) [J]. arXiv preprint arXiv:2002.06440.<br>[code:[IBM/FedMA](https://github.com/IBM/FedMA)]
- Yan Y, Niu C, Ding Y, et al. [Distributed Non-Convex Optimization with Sublinear Speedup under Intermittent Client Availability](https://arxiv.org/pdf/2002.07399)[J]. arXiv preprint arXiv:2002.07399, 2020.
- Ding Y, Niu C, Yan Y, et al. [Distributed Optimization over Block-Cyclic Data](https://arxiv.org/pdf/2002.07454)[J]. arXiv preprint arXiv:2002.07454, 2020.
- Elsa Rizk, Stefan Vlaski, Ali H. Sayed .[Dynamic Federated Learning](https://arxiv.org/pdf/2002.08782) [J]. arXiv preprint arXiv:2002.08782.
- Mher Safaryan, Egor Shulgin, Peter Richtárik .[Uncertainty Principle for Communication Compression in Distributed and Federated Learning and the Search for an Optimal Compressor](https://arxiv.org/pdf/2002.08958) [J]. arXiv preprint arXiv:2002.08958.
- Wang J, Liang H, Joshi G. [Overlap Local-SGD: An Algorithmic Approach to Hide Communication Delays in Distributed SGD](https://arxiv.org/pdf/2002.09539.pdf)[J]. arXiv preprint arXiv:2002.09539, 2020.
- Qiong Wu, Kaiwen He, Xu Chen .[Personalized Federated Learning for Intelligent IoT Applications: A Cloud-Edge based Framework](https://arxiv.org/pdf/2002.10671) [J]. arXiv preprint arXiv:2002.10671.
- Yassine Laguel, Krishna Pillutla, Jérôme Malick, Zaid Harchaoui .[Device Heterogeneity in Federated Learning: A Superquantile Approach](https://arxiv.org/pdf/2002.11223) [J]. arXiv preprint arXiv:2002.11223.
- Chen T, Sun Y, Yin W. [LASG: Lazily Aggregated Stochastic Gradients for Communication-Efficient Distributed Learning](https://arxiv.org/pdf/2002.11360)[J]. arXiv preprint arXiv:2002.11360, 2020.
- [ICML]Zhize Li, Dmitry Kovalev, Xun Qian, Peter Richtárik .[Acceleration for Compressed Gradient Descent in Distributed and Federated Optimization](https://arxiv.org/pdf/2002.11364) [J]. arXiv preprint arXiv:2002.11364.<br>[video:[v1](https://slideslive.com/38927921/acceleration-for-compressed-gradient-descent-in-distributed-optimization)]
- [Baseline]Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan .[Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295) [J]. arXiv preprint arXiv:2003.00295.
- Alekh Agarwal, John Langford, Chen-Yu Wei .[Federated Residual Learning](https://arxiv.org/pdf/2003.12880) [J]. arXiv preprint arXiv:2003.12880.
- [ICML][communication]Grigory Malinovsky, Dmitry Kovalev, Elnur Gasanov, Laurent Condat, Peter Richtarik .[From Local SGD to Local Fixed Point Methods for Federated Learning](https://arxiv.org/pdf/2004.01442) [J]. arXiv preprint arXiv:2004.01442.<br>[video:[v1](https://slideslive.com/38928320/from-local-sgd-to-local-fixed-point-methods-for-federated-learning)]
- Khanduri P, Sharma P, Kafle S, et al. [Distributed Stochastic Non-Convex Optimization: Momentum-Based Variance Reduction](https://arxiv.org/pdf/2005.00224)[J]. arXiv preprint arXiv:2005.00224, 2020.
- [NIPS][Acceleration]Reese Pathak, Martin J. Wainwright .[FedSplit: An algorithmic framework for fast federated optimization](https://arxiv.org/pdf/2005.05238) [J]. arXiv preprint arXiv:2005.05238.
- Han Cha, Jihong Park, Hyesung Kim, Mehdi Bennis, Seong-Lyun Kim .[Proxy Experience Replay: Federated Distillation for Distributed Reinforcement Learning](https://arxiv.org/pdf/2005.06105) [J]. arXiv preprint arXiv:2005.06105.
- Spiridonoff A, Olshevsky A, [Paschalidis I C. Local SGD With a Communication Overhead Depending Only on the Number of Workers](https://arxiv.org/pdf/2006.02582)[J]. arXiv preprint arXiv:2006.02582, 2020.
- Yi X, Zhang S, Yang T, et al. [A Primal-Dual SGD Algorithm for Distributed Nonconvex Optimization](https://arxiv.org/pdf/2006.03474)[J]. arXiv preprint arXiv:2006.03474, 2020.
- Shen S, Cheng Y, Liu J, et al. [STL-SGD: Speeding Up Local SGD with Stagewise Communication Period](https://arxiv.org/pdf/2006.06377)[J]. arXiv preprint arXiv:2006.06377, 2020.
- Om Thakkar, Swaroop Ramaswamy, Rajiv Mathews, Françoise Beaufays .[Understanding Unintended Memorization in Federated Learning](https://arxiv.org/pdf/2006.07490) [J]. arXiv preprint arXiv:2006.07490.
- [NIPS][Privacy]Amirhossein Reisizadeh, Farzan Farnia, Ramtin Pedarsani, Ali Jadbabaie .[Robust Federated Learning: The Case of Affine Distribution Shifts](https://arxiv.org/pdf/2006.08907) [J]. arXiv preprint arXiv:2006.08907.
- [NIPS]Honglin Yuan, Tengyu Ma .[Federated Accelerated Stochastic Gradient Descent](https://arxiv.org/pdf/2006.08950) [J]. arXiv preprint arXiv:2006.08950.<br>[code:[hongliny/FedAc-NeurIPS20](https://github.com/hongliny/FedAc-NeurIPS20)]
- Yanjie Dong, Georgios B. Giannakis, Tianyi Chen, Julian Cheng, Md. Jahangir Hossain, Victor C. M. Leung .[Communication-Efficient Robust Federated Learning Over Heterogeneous Datasets](https://arxiv.org/pdf/2006.09992) [J]. arXiv preprint arXiv:2006.09992.
- Ye T, Xiao P, Sun R. [DEED: A General Quantization Scheme for Communication Efficiency in Bits](https://arxiv.org/pdf/2006.11401)[J]. arXiv preprint arXiv:2006.11401, 2020.
- Adarsh Barik, Jean Honorio .[Exact Support Recovery in Federated Regression with One-shot Communication](https://arxiv.org/pdf/2006.12583) [J]. arXiv preprint arXiv:2006.12583.
- Thinh T. Doan .[Local Stochastic Approximation: A Unified View of Federated Learning and Distributed Multi-Task Reinforcement Learning Algorithms](https://arxiv.org/pdf/2006.13460) [J]. arXiv preprint arXiv:2006.13460.
- Charles Z, Konečný J. [On the outsized importance of learning rates in local update methods](https://arxiv.org/pdf/2007.00878)[J]. arXiv preprint arXiv:2007.00878, 2020.
- [Baseline][NIPS]Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, H. Vincent Poor .[Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/pdf/2007.07481) [J]. arXiv preprint arXiv:2007.07481.
- Farzin Haddadpour, Mohammad Mahdi Kamani, Aryan Mokhtari, Mehrdad Mahdavi .[Federated Learning with Compression: Unified Analysis and Sharp Guarantees](https://arxiv.org/pdf/2007.01154) [J]. arXiv preprint arXiv:2007.01154.
- Amani Abu Jabal, Elisa Bertino, Jorge Lobo, Dinesh Verma, Seraphin Calo, Alessandra Russo .[FLAP -- A Federated Learning Framework for Attribute-based Access Control Policies](https://arxiv.org/pdf/2010.09767) [J]. arXiv preprint arXiv:2010.09767.


## Non-IID and Model Personalization
- Takayuki Nishio, Ryo Yonetani .[Client Selection for Federated Learning with Heterogeneous Resources in  Mobile Edge](https://arxiv.org/pdf/1804.08333) [J]. arXiv preprint arXiv:1804.08333.
- Yue Zhao, Meng Li, Liangzhen Lai, Naveen Suda, Damon Civin, Vikas Chandra .[Federated Learning with Non-IID Data](https://arxiv.org/pdf/1806.00582) [J]. arXiv preprint arXiv:1806.00582.
- Eunjeong Jeong, Seungeun Oh, Hyesung Kim, Jihong Park, Mehdi Bennis, Seong-Lyun Kim .[Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data](https://arxiv.org/pdf/1811.11479) [J]. arXiv preprint arXiv:1811.11479.
- Xudong Sun, Andrea Bommert, Florian Pfisterer, Jörg Rahnenführer, Michel Lang, Bernd Bischl .[High Dimensional Restrictive Federated Model Selection with multi-objective Bayesian Optimization over shifted distributions](https://arxiv.org/pdf/1902.08999) [J]. arXiv preprint arXiv:1902.08999.
- [good]Felix Sattler, Simon Wiedemann, Klaus-Robert Müller, Wojciech Samek .[Robust and Communication-Efficient Federated Learning from Non-IID Data](https://arxiv.org/pdf/1903.02891) [J]. arXiv preprint arXiv:1903.02891.
- Yoshida N, Nishio T, Morikura M, et al. [Hybrid-FL for wireless networks: Cooperative learning mechanism using non-IID data](https://arxiv.org/pdf/1905.07210)[C]//ICC 2020-2020 IEEE International Conference on Communications (ICC). IEEE, 2020: 1-7.
- Chen X, Chen T, Sun H, et al. [Distributed training with heterogeneous data: Bridging median-and mean-based algorithms](https://arxiv.org/pdf/1906.01736.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.
- Moming Duan .[Astraea: Self-balancing Federated Learning for Improving Classification Accuracy of Mobile Deep Learning Applications](https://arxiv.org/pdf/1907.01132) [J]. arXiv preprint arXiv:1907.01132.
- [ICLR]Li X, Huang K, Yang W, et al. [On the convergence of fedavg on non-iid data](https://arxiv.org/pdf/1907.02189)[J]. arXiv preprint arXiv:1907.02189, 2019.<br>[code:[lx10077/fedavgpy](https://github.com/lx10077/fedavgpy)]
- Eunjeong Jeong, Seungeun Oh, Jihong Park, Hyesung Kim, Mehdi Bennis, Seong-Lyun Kim .[Multi-hop Federated Private Data Augmentation with Sample Compression](https://arxiv.org/pdf/1907.06426) [J]. arXiv preprint arXiv:1907.06426.
- Tzu-Ming Harry Hsu, Hang Qi, Matthew Brown .[Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/pdf/1909.06335) [J]. arXiv preprint arXiv:1909.06335.
- Guan Wang, Charlie Xiaoqian Dang, Ziye Zhou .[Measure Contribution of Participants in Federated Learning](https://arxiv.org/pdf/1909.08525) [J]. arXiv preprint arXiv:1909.08525.
- Yihan Jiang, Jakub Konečný, Keith Rush, Sreeram Kannan .[Improving Federated Learning Personalization via Model Agnostic Meta Learning](https://arxiv.org/pdf/1909.12488) [J]. arXiv preprint arXiv:1909.12488.
- Hsieh K, Phanishayee A, Mutlu O, et al. [The non-iid data quagmire of decentralized machine learning](https://arxiv.org/abs/1910.00189)[C]//International Conference on Machine Learning. PMLR, 2020: 4387-4398.
- Felix Sattler, Klaus-Robert Müller, Wojciech Samek .[Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://arxiv.org/pdf/1910.01991) [J]. arXiv preprint arXiv:1910.01991.
- Neta Shoham (Edgify), Tomer Avidor (Edgify), Aviv Keren (Edgify), Nadav Israel (Edgify), Daniel Benditkis (Edgify), Liron Mor-Yosef (Edgify), Itai Zeitak (Edgify) .[Overcoming Forgetting in Federated Learning on Non-IID Data](https://arxiv.org/pdf/1910.07796) [J]. arXiv preprint arXiv:1910.07796.
- Xin Yao, Tianchi Huang, Rui-Xiao Zhang, Ruiyu Li, Lifeng Sun .[Federated Learning with Unbiased Gradient Aggregation and Controllable Meta Updating](https://arxiv.org/pdf/1910.08234) [J]. arXiv preprint arXiv:1910.08234.
- Kangkang Wang, Rajiv Mathews, Chloé Kiddon, Hubert Eichner, Françoise Beaufays, Daniel Ramage .[Federated Evaluation of On-device Personalization](https://arxiv.org/pdf/1910.10252) [J]. arXiv preprint arXiv:1910.10252.
- [ICLR]Xingchao Peng, Zijun Huang, Yizhe Zhu, Kate Saenko .[Federated Adversarial Domain Adaptation](https://arxiv.org/pdf/1911.02054) [J]. arXiv preprint arXiv:1911.02054.
- Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, Sunav Choudhary .[Federated Learning with Personalization Layers](https://arxiv.org/pdf/1912.00818) [J]. arXiv preprint arXiv:1912.00818.
- Hesham Mostafa .[Robust Federated Learning Through Representation Matching and Adaptive Hyper-parameters](https://arxiv.org/pdf/1912.13075) [J]. arXiv preprint arXiv:1912.13075.
- Paul Pu Liang, Terrance Liu, Liu Ziyin, Ruslan Salakhutdinov, Louis-Philippe Morency .[Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/pdf/2001.01523) [J]. arXiv preprint arXiv:2001.01523.
- Sen Lin, Guang Yang, Junshan Zhang .[A Collaborative Learning Framework via Federated Meta-Learning](https://arxiv.org/pdf/2001.03229) [J]. arXiv preprint arXiv:2001.03229.
- Tiffany Tuor, Shiqiang Wang, Bong Jun Ko, Changchang Liu, Kin K. Leung .[Data Selection for Federated Learning with Relevant and Irrelevant Data at Clients](https://arxiv.org/pdf/2001.08300) [J]. arXiv preprint arXiv:2001.08300.
- Yiqiang Chen, Xiaodong Yang, Xin Qin, Han Yu, Biao Chen, Zhiqi Shen .[FOCUS: Dealing with Label Quality Disparity in Federated Learning](https://arxiv.org/pdf/2001.11359) [J]. arXiv preprint arXiv:2001.11359.
- Tao Yu, Eugene Bagdasaryan, Vitaly Shmatikov .[Salvaging Federated Learning by Local Adaptation](https://arxiv.org/pdf/2002.04758) [J]. arXiv preprint arXiv:2002.04758.
- Jia Qian, Xenofon Fafoutis, Lars Kai Hansen .[Towards Federated Learning: Robustness Analytics to Data Heterogeneity](https://arxiv.org/pdf/2002.05038) [J]. arXiv preprint arXiv:2002.05038.
- Alireza Fallah, Aryan Mokhtari, Asuman Ozdaglar .[Personalized Federated Learning: A Meta-Learning Approach](https://arxiv.org/pdf/2002.07948) [J]. arXiv preprint arXiv:2002.07948.
- Yishay Mansour, Mehryar Mohri, Jae Ro, Ananda Theertha Suresh .[Three Approaches for Personalization with Applications to Federated Learning](https://arxiv.org/pdf/2002.10619) [J]. arXiv preprint arXiv:2002.10619.
- Viraj Kulkarni, Milind Kulkarni, Aniruddha Pant .[Survey of Personalization Techniques for Federated Learning](https://arxiv.org/pdf/2003.08673) [J]. arXiv preprint arXiv:2003.08673.
- Zhikun Chen, Daofeng Li, Ming Zhao, Sihai Zhang, Jinkang Zhu .[Semi-Federated Learning](https://arxiv.org/pdf/2003.12795) [J]. arXiv preprint arXiv:2003.12795.
- Yuyang Deng, Mohammad Mahdi Kamani, Mehrdad Mahdavi .[Adaptive Personalized Federated Learning](https://arxiv.org/pdf/2003.13461) [J]. arXiv preprint arXiv:2003.13461.
- Wei Chen, Kartikeya Bhardwaj, Radu Marculescu .[FedMAX: Mitigating Activation Divergence for Accurate and Communication-Efficient Federated Learning](https://arxiv.org/pdf/2004.03657) [J]. arXiv preprint arXiv:2004.03657.
- [ICML]Felix X. Yu, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar .[Federated Learning with Only Positive Labels](https://arxiv.org/pdf/2004.10342) [J]. arXiv preprint arXiv:2004.10342.<br>[video:[federated-learning-with-only-positive-labels](https://slideslive.com/38928322/federated-learning-with-only-positive-labels)]
- Christopher Briggs, Zhong Fan, Peter Andras .[Federated learning with hierarchical clustering of local updates to improve training on non-IID data](https://arxiv.org/pdf/2004.11791) [J]. arXiv preprint arXiv:2004.11791.
- Ming Xie, Guodong Long, Tao Shen, Tianyi Zhou, Xianzhi Wang, Jing Jiang .[Multi-Center Federated Learning](https://arxiv.org/pdf/2005.01026) [J]. arXiv preprint arXiv:2005.01026.
- Han Cha, Jihong Park, Hyesung Kim, Mehdi Bennis, Seong-Lyun Kim .[Proxy Experience Replay: Federated Distillation for Distributed Reinforcement Learning](https://arxiv.org/pdf/2005.06105) [J]. arXiv preprint arXiv:2005.06105.
- Ahn S, Ozgur A, Pilanci M. [Global Multiclass Classification from Heterogeneous Local Models](https://arxiv.org/pdf/2005.10848)[J]. arXiv preprint arXiv:2005.10848, 2020.
- Xinwei Zhang, Mingyi Hong, Sairaj Dhople, Wotao Yin, Yang Liu .[FedPD: A Federated Learning Framework with Optimal Rates and Adaptivity to Non-IID Data](https://arxiv.org/pdf/2005.11418) [J]. arXiv preprint arXiv:2005.11418.
- Cong Wang, Yuanyuan Yang, Pengzhan Zhou .[Towards Efficient Scheduling of Federated Mobile Devices under Computational and Statistical Heterogeneity](https://arxiv.org/pdf/2005.12326) [J]. arXiv preprint arXiv:2005.12326.
- Xin Yao, Lifeng Sun .[Continual Local Training for Better Initialization of Federated Models](https://arxiv.org/pdf/2005.12657) [J]. arXiv preprint arXiv:2005.12657.
- [NIPS]Avishek Ghosh, Jichan Chung, Dong Yin, Kannan Ramchandran .[An Efficient Framework for Clustered Federated Learning](https://arxiv.org/pdf/2006.04088) [J]. arXiv preprint arXiv:2006.04088.
- MyungJae Shin, Chihoon Hwang, Joongheon Kim, Jihong Park, Mehdi Bennis, Seong-Lyun Kim .[XOR Mixup: Privacy-Preserving Data Augmentation for One-Shot Federated Learning](https://arxiv.org/pdf/2006.05148) [J]. arXiv preprint arXiv:2006.05148.
- Yichen Ruan, Xiaoxi Zhang, Shu-Che Liang, Carlee Joe-Wong .[Towards Flexible Device Participation in Federated Learning for Non-IID Data](https://arxiv.org/pdf/2006.06954) [J]. arXiv preprint arXiv:2006.06954.
- [NIPS]Tao Lin, Lingjing Kong, Sebastian U. Stich, Martin Jaggi .[Ensemble Distillation for Robust Model Fusion in Federated Learning](https://arxiv.org/pdf/2006.07242) [J]. arXiv preprint arXiv:2006.07242.
- [NIPS]Canh T. Dinh, Nguyen H. Tran, Tuan Dung Nguyen .[Personalized Federated Learning with Moreau Envelopes](https://arxiv.org/pdf/2006.08848) [J]. arXiv preprint arXiv:2006.08848.<br>[code:[CharlieDinh/pFedMe](https://github.com/CharlieDinh/pFedMe)]
- [NIPS][Privacy]Amirhossein Reisizadeh, Farzan Farnia, Ramtin Pedarsani, Ali Jadbabaie .[Robust Federated Learning: The Case of Affine Distribution Shifts](https://arxiv.org/pdf/2006.08907) [J]. arXiv preprint arXiv:2006.08907.
- Kavya Kopparapu, Eric Lin, Jessica Zhao .[FedCD: Improving Performance in non-IID Federated Learning](https://arxiv.org/pdf/2006.09637) [J]. arXiv preprint arXiv:2006.09637.
- Kavya Kopparapu, Eric Lin .[FedFMC: Sequential Efficient Federated Learning on Non-iid Data](https://arxiv.org/pdf/2006.10937) [J]. arXiv preprint arXiv:2006.10937.
- Wonyong Jeong, Jaehong Yoon, Eunho Yang, Sung Ju Hwang .[Federated Semi-Supervised Learning with Inter-Client Consistency](https://arxiv.org/pdf/2006.12097) [J]. arXiv preprint arXiv:2006.12097.
- Laura Rieger, Rasmus M. Th. Høegh, Lars K. Hansen .[Client Adaptation improves Federated Learning with Simulated Non-IID Clients](https://arxiv.org/pdf/2007.04806) [J]. arXiv preprint arXiv:2007.04806.


## Semi-Supervised Learning
- Papernot N, Abadi M, Erlingsson U, et al. [Semi-supervised knowledge transfer for deep learning from private training data](https://arxiv.org/pdf/1610.05755.pdf,)[J]. arXiv preprint arXiv:1610.05755, 2016.
- Papernot N, Song S, Mironov I, et al. [Scalable private learning with pate](https://arxiv.org/pdf/1802.08908.pdf?ref=hackernoon.com)[J]. arXiv preprint arXiv:1802.08908, 2018.
- Wonyong Jeong, Jaehong Yoon, Eunho Yang, Sung Ju Hwang .[Federated Semi-Supervised Learning with Inter-Client Consistency](https://arxiv.org/pdf/2006.12097) [J]. arXiv preprint arXiv:2006.12097.


## Vertical Federated Learning
- [LM][CM]A. P. Sanil, A. F. Karr, X. Lin, and J. P. Reiter, "Privacy preserving regression modelling via distributed computation," in Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2004, pp. 677–682.
- [LM][CM]Stephen Hardy, Wilko Henecka, Hamish Ivey-Law, Richard Nock, Giorgio Patrini, Guillaume Smith, Brian Thorne .[Private federated learning on vertically partitioned data via entity  resolution and additively homomorphic encryption](https://arxiv.org/pdf/1711.10677) [J]. arXiv preprint arXiv:1711.10677.
-  .[Entity Resolution and Federated Learning get a Federated Resolution](https://arxiv.org/pdf/1803.04035) [J]. arXiv preprint arXiv:1803.04035.
- [NN][CM]Y. Liu, T. Chen, and Q. Yang, [Secure federated transfer learning](https://arxiv.org/pdf/1812.03337)[J] arXiv preprint arXiv:1812.03337, 2018.
- [DT][CM]Kewei Cheng, Tao Fan, Yilun Jin, Yang Liu, Tianjian Chen, Qiang Yang .[SecureBoost: A Lossless Federated Learning Framework](https://arxiv.org/pdf/1901.08755) [J]. arXiv preprint arXiv:1901.08755.
- Shengwen Yang, Bing Ren, Xuhui Zhou, Liping Liu .[Parallel Distributed Logistic Regression for Vertical Federated Learning without Third-Party Coordinator](https://arxiv.org/pdf/1911.09824) [J]. arXiv preprint arXiv:1911.09824.
- Kai Yang, Tao Fan, Tianjian Chen, Yuanming Shi, Qiang Yang .[A Quasi-Newton Method Based Vertical Federated Learning Framework for Logistic Regression](https://arxiv.org/pdf/1912.00513) [J]. arXiv preprint arXiv:1912.00513.
- Yang Liu, Yan Kang, Xinwei Zhang, Liping Li, Yong Cheng, Tianjian Chen, Mingyi Hong, Qiang Yang .[A Communication Efficient Vertical Federated Learning Framework](https://arxiv.org/pdf/1912.11187) [J]. arXiv preprint arXiv:1912.11187.
- Siwei Feng, Han Yu .[Multi-Participant Multi-Class Vertical Federated Learning](https://arxiv.org/pdf/2001.11154) [J]. arXiv preprint arXiv:2001.11154.
- Yang Liu, Xiong Zhang, Libin Wang .[Asymmetrical Vertical Federated Learning](https://arxiv.org/pdf/2004.07427) [J]. arXiv preprint arXiv:2004.07427.
- Tianyi Chen, Xiao Jin, Yuejiao Sun, Wotao Yin .[VAFL: a Method of Vertical Asynchronous Federated Learning](https://arxiv.org/pdf/2007.06081) [J]. arXiv preprint arXiv:2007.06081.


## Hierarchical Federated Learning && Horizontal Federated Learning
- [NN][MA]H. Zhu and Y. Jin, "Multi-objective evolutionary federated learning," IEEE transactions on neural networks and learning systems, 2019.
- [NN][GAN]A. Triastcyn and B. Faltings, "Federated generative privacy," 2019.
- [NN][CM,MA]K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, and K. Seth, "Practical secure aggregation for privacy-preserving machine learning," in Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security. ACM, 2017, pp. 1175–1191. 
- [NN][DP,MA]R. Shokri and V. Shmatikov, "Privacy-preserving deep learning," in Proceedings of the 22nd ACM SIGSAC conference on computer and communications security. ACM, 2015, pp. 1310–1321.
- [NN][MA]P. Blanchard, R. Guerraoui, J. Stainer et al., "Machine learning with adversaries: Byzantine tolerant gradient descent," in Advances in Neural Information Processing Systems, 2017, pp. 119–129. 
- [LM][CM]V. Nikolaenko, U. Weinsberg, S. Ioannidis, M. Joye, D. Boneh, and N. Taft, "Privacy-preserving ridge regression on hundreds of millions of records," in 2013 IEEE Symposium on Security and Privacy. IEEE, 2013, pp. 334–348.
- [LM][MA]Y. Chen, L. Su, and J. Xu, "Distributed statistical machine learning in adversarial settings: Byzantine gradient descent," Proceedings of the ACM on Measurement and Analysis of Computing Systems, vol. 1, no. 2, p. 44, 2017.
- [LM][MA]V. Smith, C.-K. Chiang, M. Sanjabi, and A. S. Talwalkar, "Federated multi-task learning," in Advances in Neural Information Processing Systems, 2017, pp. 4424–4434. 
- [DT][DP]L. Zhao, L. Ni, S. Hu, Y. Chen, P. Zhou, F. Xiao, and L. Wu, "Inprivate digging: Enabling tree-based distributed data mining with differential privacy," in INFOCOM. IEEE, 2018, pp. 2087–2095.
- [LM][CM]Y.-R. Chen, A. Rezapour, and W.-G. Tzeng, "Privacy-preserving ridge regression on distributed data," Information Sciences, vol. 451, pp. 34–49, 2018.
- [NN][MA]G. Ulm, E. Gustavsson, and M. Jirstrand, "Functional federated learning in erlang (ffl-erl)," in International Workshop on Functional and Constraint Logic Programming. Springer, 2018, pp. 162–178. 
- [NN][MA]T. Nishio and R. Yonetani, "Client selection for federated learning with heterogeneous resources in mobile edge," in ICC 2019-2019 IEEE International Conference on Communications (ICC). IEEE, 2019, pp. 1–7.
- [NN][MA]X. Wang, Y. Han, C. Wang, Q. Zhao, X. Chen, and M. Chen, "In-edge ai: Intelligentizing mobile edge computing, caching and communication by federated learning," IEEE Network, 2019.
- [LM,NN][MA]S. Wang, T. Tuor, T. Salonidis, K. K. Leung, C. Makaya, T. He, and K. Chan, "Adaptive federated learning in resource constrained edge computing systems," IEEE Journal on Selected Areas in Communications, vol. 37, no. 6, pp. 1205–1221, 2019.
- [NN][MA]A. Nilsson, S. Smith, G. Ulm, E. Gustavsson, and M. Jirstrand, "A performance evaluation of federated learning algorithms," in Proceedings of the Second Workshop on Distributed Infrastructures for Deep Learning. ACM, 2018, pp. 1–8. 
- [NN][MA][Baseline]Brendan McMahan H, Moore E, Ramage D, et al. [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)[J]. arXiv, 2016: arXiv: 1602.05629.
- [LM][MA]Jakub Konečný, H. Brendan McMahan, Daniel Ramage, Peter Richtárik .[Federated Optimization: Distributed Machine Learning for On-Device  Intelligence](https://arxiv.org/pdf/1610.02527) [J]. arXiv preprint arXiv:1610.02527.
- [NN][MA]Jakub Konečný, H. Brendan McMahan, Felix X. Yu, Peter Richtárik, Ananda Theertha Suresh, Dave Bacon .[Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/pdf/1610.05492) [J]. arXiv preprint arXiv:1610.05492.
- [LM,DT,NN][DP,MA]Papernot N, Abadi M, Erlingsson U, et al. [Semi-supervised knowledge transfer for deep learning from private training data](https://arxiv.org/pdf/1610.05755.pdf,)[J]. arXiv preprint arXiv:1610.05755, 2016.
- [NN][DP,MA]McMahan H B, Ramage D, Talwar K, et al. [Learning differentially private recurrent language models](https://arxiv.org/pdf/1710.06963)[J]. arXiv preprint arXiv:1710.06963, 2017.
- [NN][DP,MA]Robin C. Geyer, Tassilo Klein, Moin Nabi .[Differentially Private Federated Learning: A Client Level Perspective](https://arxiv.org/pdf/1712.07557) [J]. arXiv preprint arXiv:1712.07557.
- [NN][MA]Fei Chen, Zhenhua Dong, Zhenguo Li, Xiuqiang He .[Federated Meta-Learning for Recommendation](https://arxiv.org/pdf/1802.07876) [J]. arXiv preprint arXiv:1802.07876.
- [GPD][MA][]Sumudu Samarakoon, Mehdi Bennis, Walid Saad, Merouane Debbah .[Distributed Federated Learning for Ultra-Reliable Low-Latency Vehicular  Communications](https://arxiv.org/pdf/1807.08127) [J]. arXiv preprint arXiv:1807.08127.
- [LM][MA]Hyesung Kim, Jihong Park, Mehdi Bennis, Seong-Lyun Kim .[On-Device Federated Learning via Blockchain and its Latency Analysis](https://arxiv.org/pdf/1808.03949) [J]. arXiv preprint arXiv:1808.03949.
- [NN][MA]Andrew Hard, Kanishka Rao, Rajiv Mathews, Françoise Beaufays, Sean Augenstein, Hubert Eichner, Chloé Kiddon, Daniel Ramage .[Federated Learning for Mobile Keyboard Prediction](https://arxiv.org/pdf/1811.03604) [J]. arXiv preprint arXiv:1811.03604.
- [NN][MA]Eunjeong Jeong, Seungeun Oh, Hyesung Kim, Jihong Park, Mehdi Bennis, Seong-Lyun Kim .[Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data](https://arxiv.org/pdf/1811.11479) [J]. arXiv preprint arXiv:1811.11479.
- [LM,NN][MA]Abhishek Bhowmick, John Duchi, Julien Freudiger, Gaurav Kapoor, Ryan Rogers .[Protection Against Reconstruction and Its Applications in Private Federated Learning](https://arxiv.org/pdf/1812.00984) [J]. arXiv preprint arXiv:1812.00984.
- [LM,DT,NN][CM,DP,MA]S. Truex, N. Baracaldo, A. Anwar, T. Steinke, H. Ludwig, and R. Zhang, [A hybrid approach to privacy-preserving federated learning](https://arxiv.org/pdf/1812.03224)[J] arXiv preprint arXiv:1812.03224, 2018. 
- [LM][MA]Muhammad Ammad-ud-din, Elena Ivannikova, Suleiman A. Khan, Were Oyomno, Qiang Fu, Kuan Eeik Tan, Adrian Flanagan .[Federated Collaborative Filtering for Privacy-Preserving Personalized Recommendation System](https://arxiv.org/pdf/1901.09888) [J]. arXiv preprint arXiv:1901.09888.
- [NN][MA]Boyi Liu, Lujia Wang, Ming Liu, Chengzhong Xu .[Lifelong Federated Reinforcement Learning: A Learning Architecture for Navigation in Cloud Robotic Systems](https://arxiv.org/pdf/1901.06455) [J]. arXiv preprint arXiv:1901.06455.
- [LM,NN][MA][ICML]Mehryar Mohri, Gary Sivek, Ananda Theertha Suresh .[Agnostic Federated Learning](https://arxiv.org/pdf/1902.00146) [J]. arXiv preprint arXiv:1902.00146.
- [NN][MA]Felix Sattler, Simon Wiedemann, Klaus-Robert Müller, Wojciech Samek .[Robust and Communication-Efficient Federated Learning from Non-IID Data](https://arxiv.org/pdf/1903.02891) [J]. arXiv preprint arXiv:1903.02891.
- Lumin Liu, Jun Zhang, S. H. Song, Khaled B. Letaief .[Edge-Assisted Hierarchical Federated Learning with Non-IID Data](https://arxiv.org/pdf/1905.06641) [J]. arXiv preprint arXiv:1905.06641.
- [LM,NN][MA][ICLR]Tian Li, Maziar Sanjabi, Virginia Smith .[Fair Resource Allocation in Federated Learning](https://arxiv.org/pdf/1905.10497) [J]. arXiv preprint arXiv:1905.10497.<br>[code:[litian96/fair_flearn](https://github.com/litian96/fair_flearn)]
- [NN][MA][ICML]Mikhail Yurochkin, Mayank Agarwal, Soumya Ghosh, Kristjan Greenewald, Trong Nghia Hoang, Yasaman Khazaeni .[Bayesian Nonparametric Federated Learning of Neural Networks](https://arxiv.org/pdf/1905.12022) [J]. arXiv preprint arXiv:1905.12022.<br>[code:[IBM/probabilistic-federated-neural-matching](https://github.com/IBM/probabilistic-federated-neural-matching)]
- Wang J, Sahu A K, Yang Z, et al. [MATCHA: Speeding up decentralized SGD via matching decomposition sampling](https://arxiv.org/pdf/1905.09435)[J]. arXiv preprint arXiv:1905.09435, 2019.
- Feng Liao, Hankz Hankui Zhuo, Xiaoling Huang, Yu Zhang .[Federated Hierarchical Hybrid Networks for Clickbait Detection](https://arxiv.org/pdf/1906.00638) [J]. arXiv preprint arXiv:1906.00638.
- [NN][MA]Luca Corinzia, Joachim M. Buhmann .[Variational Federated Multi-Task Learning](https://arxiv.org/pdf/1906.06268) [J]. arXiv preprint arXiv:1906.06268.
- [DT][CM]Yang Liu, Zhuo Ma, Ximeng Liu, Siqi Ma, Surya Nepal, Robert Deng .[Boosting Privately: Privacy-Preserving Federated Extreme Boosting for Mobile Crowdsensing](https://arxiv.org/pdf/1907.10218) [J]. arXiv preprint arXiv:1907.10218.
- Mehdi Salehi Heydar Abad, Emre Ozfatura, Deniz Gunduz, Ozgur Ercetin .[Hierarchical Federated Learning Across Heterogeneous Cellular Networks](https://arxiv.org/pdf/1909.02362) [J]. arXiv preprint arXiv:1909.02362.
- [DT][Hash]Qinbin Li, Zeyi Wen, Bingsheng He .[Practical Federated Gradient Boosting Decision Trees](https://arxiv.org/pdf/1911.04206) [J]. arXiv preprint arXiv:1911.04206.
- Li H, Meng D, Li X. [Knowledge Federation: Hierarchy and Unification](https://arxiv.org/pdf/2002.01647)[J]. arXiv preprint arXiv:2002.01647, 2020.
- Siqi Luo, Xu Chen, Qiong Wu, Zhi Zhou, Shuai Yu .[HFEL: Joint Edge Association and Resource Allocation for Cost-Efficient Hierarchical Federated Edge Learning](https://arxiv.org/pdf/2002.11343) [J]. arXiv preprint arXiv:2002.11343.
- Aidmar Wainakh, Alejandro Sanchez Guinea, Tim Grube, Max Mühlhäuser .[Enhancing Privacy via Hierarchical Federated Learning](https://arxiv.org/pdf/2004.11361) [J]. arXiv preprint arXiv:2004.11361.
- Christopher Briggs, Zhong Fan, Peter Andras .[Federated learning with hierarchical clustering of local updates to improve training on non-IID data](https://arxiv.org/pdf/2004.11791) [J]. arXiv preprint arXiv:2004.11791.


## Decentralized Federated Learning
- Lian X, Zhang C, Zhang H, et al. [Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent](https://arxiv.org/pdf/1705.09056.pdf)[C]//Advances in Neural Information Processing Systems. 2017: 5330-5340.
- Shayan M, Fung C, Yoon C J M, et al. [Biscotti: A ledger for private and secure peer-to-peer machine learning](https://arxiv.org/pdf/1811.09904)[J]. arXiv preprint arXiv:1811.09904, 2018.
- Abhijit Guha Roy, Shayan Siddiqui, Sebastian Pölsterl, Nassir Navab, Christian Wachinger .[BrainTorrent: A Peer-to-Peer Environment for Decentralized Federated Learning](https://arxiv.org/pdf/1905.06731) [J]. arXiv preprint arXiv:1905.06731.
- Wang J, Sahu A K, Yang Z, et al. [MATCHA: Speeding up decentralized SGD via matching decomposition sampling](https://arxiv.org/pdf/1905.09435)[J]. arXiv preprint arXiv:1905.09435, 2019.
- Lalitha A, Wang X, Kilinc O, et al. [Decentralized bayesian learning over graphs](https://arxiv.org/pdf/1905.10466)[J]. arXiv preprint arXiv:1905.10466, 2019.
- Chaoyang He, Conghui Tan, Hanlin Tang, Shuang Qiu, Ji Liu .[Central Server Free Federated Learning over Single-sided Trust Social Networks](https://arxiv.org/pdf/1910.04956) [J]. arXiv preprint arXiv:1910.04956.
- Ye H, Luo L, Zhou Z, et al. [Multi-consensus Decentralized Accelerated Gradient Descent](https://arxiv.org/pdf/2005.00797)[J]. arXiv preprint arXiv:2005.00797, 2020.


## Federated Transfer Learning
- Eunjeong Jeong, Seungeun Oh, Hyesung Kim, Jihong Park, Mehdi Bennis, Seong-Lyun Kim .[Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data](https://arxiv.org/pdf/1811.11479) [J]. arXiv preprint arXiv:1811.11479.
- [good]Yang Liu, Tianjian Chen, Qiang Yang .[Secure Federated Transfer Learning](https://arxiv.org/pdf/1812.03337) [J]. arXiv preprint arXiv:1812.03337.
- Jin-Hyun Ahn, Osvaldo Simeone, Joonhyuk Kang .[Wireless Federated Distillation for Distributed Edge Learning with Heterogeneous Data](https://arxiv.org/pdf/1907.02745) [J]. arXiv preprint arXiv:1907.02745.
- Han Cha, Jihong Park, Hyesung Kim, Seong-Lyun Kim, Mehdi Bennis .[Federated Reinforcement Distillation with Proxy Experience Memory](https://arxiv.org/pdf/1907.06536) [J]. arXiv preprint arXiv:1907.06536.
- Daliang Li, Junpu Wang .[FedMD: Heterogenous Federated Learning via Model Distillation](https://arxiv.org/pdf/1910.03581) [J]. arXiv preprint arXiv:1910.03581.
- Shreya Sharma, Xing Chaoping, Yang Liu, Yan Kang .[Secure and Efficient Federated Transfer Learning](https://arxiv.org/pdf/1910.13271) [J]. arXiv preprint arXiv:1910.13271.
- Chang H, Shejwalkar V, Shokri R, et al. [Cronus: Robust and Heterogeneous Collaborative Learning with Black-Box Knowledge Transfer](https://arxiv.org/pdf/1912.11279)[J]. arXiv preprint arXiv:1912.11279, 2019.
- Jin-Hyun Ahn, Osvaldo Simeone, Joonhyuk Kang .[Cooperative Learning via Federated Distillation over Fading Channels](https://arxiv.org/pdf/2002.01337) [J]. arXiv preprint arXiv:2002.01337.
- Li H, Meng D, Li X. [Knowledge Federation: Hierarchy and Unification](https://arxiv.org/pdf/2002.01647)[J]. arXiv preprint arXiv:2002.01647, 2020.
- Fay D, Sjölund J, Oechtering T J. [Decentralized Differentially Private Segmentation with PATE](https://arxiv.org/pdf/2004.06567)[J]. arXiv preprint arXiv:2004.06567, 2020.
- Han Cha, Jihong Park, Hyesung Kim, Mehdi Bennis, Seong-Lyun Kim .[Proxy Experience Replay: Federated Distillation for Distributed Reinforcement Learning](https://arxiv.org/pdf/2005.06105) [J]. arXiv preprint arXiv:2005.06105.


## Neural Architecture Search
- Xu M, Zhao Y, Bian K, et al. [Neural Architecture Search over Decentralized Data](https://arxiv.org/pdf/2002.06352)[J]. arXiv preprint arXiv:2002.06352, 2020.
- Hangyu Zhu, Yaochu Jin .[Real-time Federated Evolutionary Neural Architecture Search](https://arxiv.org/pdf/2003.02793) [J]. arXiv preprint arXiv:2003.02793.
- Chaoyang He, Murali Annavaram, Salman Avestimehr .[FedNAS: Federated Deep Learning via Neural Architecture Search](https://arxiv.org/pdf/2004.08546) [J]. arXiv preprint arXiv:2004.08546.
- Ishika Singh, Haoyi Zhou, Kunlin Yang, Meng Ding, Bill Lin, Pengtao Xie .[Differentially-private Federated Neural Architecture Search](https://arxiv.org/pdf/2006.10559) [J]. arXiv preprint arXiv:2006.10559.
- Hangyu Zhu, Haoyu Zhang, Yaochu Jin .[From Federated Learning to Federated Neural Architecture Search: A Survey](https://arxiv.org/pdf/2009.05868) [J]. arXiv preprint arXiv:2009.05868.
- Anubhav Garg, Amit Kumar Saha, Debo Dutta .[Direct Federated Neural Architecture Search](https://arxiv.org/pdf/2010.06223) [J]. arXiv preprint arXiv:2010.06223.


## Continual Learning
- Jaehong Yoon, Wonyong Jeong, Giwoong Lee, Eunho Yang, Sung Ju Hwang .[Federated Continual Learning with Adaptive Parameter Communication](https://arxiv.org/pdf/2003.03196) [J]. arXiv preprint arXiv:2003.03196.


## Reinforcement Learning && Robotics
- Luong, Nguyen Cong, et al. [Efficient Training Management for Mobile Crowd-Machine Learning: A Deep Reinforcement Learning Approach.](https://arxiv.org/pdf/1812.03633) IEEE Wireless Communications Letters, vol. 8, no. 5, 2019, pp. 1345–1348.
- Boyi Liu, Lujia Wang, Ming Liu, Chengzhong Xu .[Lifelong Federated Reinforcement Learning: A Learning Architecture for Navigation in Cloud Robotic Systems](https://arxiv.org/pdf/1901.06455) [J]. arXiv preprint arXiv:1901.06455.
- [good]Hankz Hankui Zhuo, Wenfeng Feng, Qian Xu, Qiang Yang, Yufeng Lin .[Federated Reinforcement Learning](https://arxiv.org/pdf/1901.08277) [J]. arXiv preprint arXiv:1901.08277.
- Boyi Liu, Lujia Wang, Ming Liu, Cheng-Zhong Xu .[Federated Imitation Learning: A Privacy Considered Imitation Learning Framework for Cloud Robotic Systems with Heterogeneous Sensor Data](https://arxiv.org/pdf/1909.00895) [J]. arXiv preprint arXiv:1909.00895.
- Haozhao Wang, Zhihao Qu, Song Guo, Xin Gao, Ruixuan Li, Baoliu Ye .[Intermittent Pulling with Local Compensation for Communication-Efficient Federated Learning](https://arxiv.org/pdf/2001.08277) [J]. arXiv preprint arXiv:2001.08277.


## Bayesian Learning
- Xudong Sun, Andrea Bommert, Florian Pfisterer, Jörg Rahnenführer, Michel Lang, Bernd Bischl .[High Dimensional Restrictive Federated Model Selection with multi-objective Bayesian Optimization over shifted distributions](https://arxiv.org/pdf/1902.08999) [J]. arXiv preprint arXiv:1902.08999.
- Mrinank Sharma, Michael Hutchinson, Siddharth Swaroop, Antti Honkela, Richard E. Turner .[Differentially Private Federated Variational Inference](https://arxiv.org/pdf/1911.10563) [J]. arXiv preprint arXiv:1911.10563.


## Adversarial-Attack-and-Defense
- Hitaj B, Ateniese G, Perez-Cruz F. [Deep models under the GAN: information leakage from collaborative deep learning](https://arxiv.org/pdf/1702.07464.pdf)[C]//Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security. 2017: 603-618.
- [ICLR]Xie C, Huang K, Chen P Y, et al. [DBA: Distributed Backdoor Attacks against Federated Learning](http://www.openreview.net/pdf?id=rkgyS0VFvr)[C]//International Conference on Learning Representations. 2019.<br>[code:[AI-secure/DBA](https://github.com/AI-secure/DBA)]
- Yin D, Chen Y, Ramchandran K, et al. [Byzantine-robust distributed learning: Towards optimal statistical rates](https://arxiv.org/pdf/1803.01498)[J]. arXiv preprint arXiv:1803.01498, 2018.
- Melis L, Song C, De Cristofaro E, et al. [Exploiting unintended feature leakage in collaborative learning](https://arxiv.org/pdf/1805.04049)[C]//2019 IEEE Symposium on Security and Privacy (SP). IEEE, 2019: 691-706. <br>[code:[csong27/property-inference-collaborative-ml](https://github.com/csong27/property-inference-collaborative-ml)]
- [good]Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, Vitaly Shmatikov .[How To Backdoor Federated Learning](https://arxiv.org/pdf/1807.00459) [J]. arXiv preprint arXiv:1807.00459.<br>[code:[ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)]
- Clement Fung, Chris J.M. Yoon, Ivan Beschastnikh .[Mitigating Sybils in Federated Learning Poisoning](https://arxiv.org/pdf/1808.04866) [J]. arXiv preprint arXiv:1808.04866.
- Li L, Xu W, Chen T, et al. [RSA: Byzantine-robust stochastic aggregation methods for distributed learning from heterogeneous datasets](https://www.aaai.org/ojs/index.php/AAAI/article/view/3968/3846)[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33: 1544-1551. 
- Fung C, Koerner J, Grant S, et al. [Dancing in the dark: private multi-party machine learning in an untrusted setting](https://arxiv.org/pdf/1811.09712)[J]. arXiv preprint arXiv:1811.09712, 2018.
- [ICML]Arjun Nitin Bhagoji, Supriyo Chakraborty, Prateek Mittal, Seraphin Calo .[Analyzing Federated Learning through an Adversarial Lens](https://arxiv.org/pdf/1811.12470) [J]. arXiv preprint arXiv:1811.12470.<br>[code:[inspire-group/ModelPoisoning](https://github.com/inspire-group/ModelPoisoning)]
- Zhibo Wang, Mengkai Song, Zhifei Zhang, Yang Song, Qian Wang, Hairong Qi .[Beyond Inferring Class Representatives: User-Level Privacy Leakage From Federated Learning](https://arxiv.org/pdf/1812.00535) [J]. arXiv preprint arXiv:1812.00535.
- Milad Nasr, Reza Shokri, Amir Houmansadr .[Comprehensive Privacy Analysis of Deep Learning: Stand-alone and Federated Learning under Passive and Active White-box Inference Attacks](https://arxiv.org/pdf/1812.00910) [J]. arXiv preprint arXiv:1812.00910.
- Abhishek Bhowmick, John Duchi, Julien Freudiger, Gaurav Kapoor, Ryan Rogers .[Protection Against Reconstruction and Its Applications in Private Federated Learning](https://arxiv.org/pdf/1812.00984) [J]. arXiv preprint arXiv:1812.00984.
- Chen, Qingrong, et al. [Differentially Private Data Generative Models.](https://arxiv.org/pdf/1812.02274) ArXiv Preprint ArXiv:1812.02274, 2018.
- Zhu L, Liu Z, Han S. [Deep leakage from gradients](https://papers.nips.cc/paper/9617-deep-leakage-from-gradients.pdf)[C]//Advances in Neural Information Processing Systems. 2019: 14774-14784.
- [NIPS]Baruch, Moran, et al. [A Little Is Enough: Circumventing Defenses For Distributed Learning.](https://arxiv.org/pdf/1902.06156) ArXiv Preprint ArXiv:1902.06156, 2019.
- [AAAI]Yufei Han, Xiangliang Zhang .[Robust Federated Training via Collaborative Machine Teaching using Trusted Instances](https://arxiv.org/pdf/1905.02941) [J]. arXiv preprint arXiv:1905.02941.
- Dong Y, Cheng J, Hossain M J, et al. [Secure distributed on-device learning networks with Byzantine adversaries](https://arxiv.org/pdf/1906.00887)[J]. IEEE Network, 2019, 33(6): 180-187.
- Yang Liu, Zhuo Ma, Ximeng Liu, Siqi Ma, Surya Nepal, Robert Deng .[Boosting Privately: Privacy-Preserving Federated Extreme Boosting for Mobile Crowdsensing](https://arxiv.org/pdf/1907.10218) [J]. arXiv preprint arXiv:1907.10218.
- Hongyu Li, Tianqi Han .[An End-to-End Encrypted Neural Network for Gradient Updates Transmission in Federated Learning](https://arxiv.org/pdf/1908.08340) [J]. arXiv preprint arXiv:1908.08340.
- Luis Muñoz-González, Kenneth T. Co, Emil C. Lupu .[Byzantine-Robust Federated Machine Learning through Adaptive Model Averaging](https://arxiv.org/pdf/1909.05125) [J]. arXiv preprint arXiv:1909.05125.
- Zhaorui Li, Zhicong Huang, Chaochao Chen, Cheng Hong .[Quantification of the Leakage in Federated Learning](https://arxiv.org/pdf/1910.05467) [J]. arXiv preprint arXiv:1910.05467.
- Lixu Wang, Shichao Xu, Xiao Wang, Qi Zhu .[Eavesdrop the Composition Proportion of Training Labels in Federated Learning](https://arxiv.org/pdf/1910.06044) [J]. arXiv preprint arXiv:1910.06044.
- Suyi Li, Yong Cheng, Yang Liu, Wei Wang, Tianjian Chen .[Abnormal Client Behavior Detection in Federated Learning](https://arxiv.org/pdf/1910.09933) [J]. arXiv preprint arXiv:1910.09933.
- Fan Ang, Li Chen, Nan Zhao, Yunfei Chen, Weidong Wang, F. Richard Yu .[Robust Federated Learning with Noisy Communication](https://arxiv.org/pdf/1911.00251) [J]. arXiv preprint arXiv:1911.00251.
- [good]Ziteng Sun, Peter Kairouz, Ananda Theertha Suresh, H. Brendan McMahan .[Can You Really Backdoor Federated Learning?](https://arxiv.org/pdf/1911.07963) [J]. arXiv preprint arXiv:1911.07963.
- Minghong Fang, Xiaoyu Cao, Jinyuan Jia, Neil Zhenqiang Gong .[Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](https://arxiv.org/pdf/1911.11815) [J]. arXiv preprint arXiv:1911.11815.
- Jierui Lin, Min Du, Jian Liu .[Free-riders in Federated Learning: Attacks and Defenses](https://arxiv.org/pdf/1911.12560) [J]. arXiv preprint arXiv:1911.12560.
- Chang H, Shejwalkar V, Shokri R, et al. [Cronus: Robust and Heterogeneous Collaborative Learning with Black-Box Knowledge Transfer](https://arxiv.org/pdf/1912.11279)[J]. arXiv preprint arXiv:1912.11279, 2019.
- Shuhao Fu, Chulin Xie, Bo Li, Qifeng Chen .[Attack-Resistant Federated Learning with Residual-based Reweighting](https://arxiv.org/pdf/1912.11464) [J]. arXiv preprint arXiv:1912.11464.
- Josh Payne, Ashish Kundu .[Towards Deep Federated Defenses Against Malware in Cloud Ecosystems](https://arxiv.org/pdf/1912.12370) [J]. arXiv preprint arXiv:1912.12370.
- Krishna Pillutla, Sham M. Kakade, Zaid Harchaoui .[Robust Aggregation for Federated Learning](https://arxiv.org/pdf/1912.13445) [J]. arXiv preprint arXiv:1912.13445.
- Suyi Li, Yong Cheng, Wei Wang, Yang Liu, Tianjian Chen .[Learning to Detect Malicious Clients for Robust Federated Learning](https://arxiv.org/pdf/2002.00211) [J]. arXiv preprint arXiv:2002.00211.
- Richeng Jin, Yufan Huang, Xiaofan He, Huaiyu Dai, Tianfu Wu .[Stochastic-Sign SGD for Federated Learning with Theoretical Guarantees](https://arxiv.org/pdf/2002.10940) [J]. arXiv preprint arXiv:2002.10940.
- Yang Y R, Li W J. [BASGD: Buffered Asynchronous SGD for Byzantine Learning](https://arxiv.org/pdf/2003.00937)[J]. arXiv preprint arXiv:2003.00937, 2020.
- Huafei Zhu, Zengxiang Li, Merivyn Cheah, Rick Siow Mong Goh .[Privacy-preserving Weighted Federated Learning within Oracle-Aided MPC Framework](https://arxiv.org/pdf/2003.07630) [J]. arXiv preprint arXiv:2003.07630.
- Rui Hu, Yuanxiong Guo, E. Paul. Ratazzi, Yanmin Gong .[Differentially Private Federated Learning for Resource-Constrained Internet of Things](https://arxiv.org/pdf/2003.12705) [J]. arXiv preprint arXiv:2003.12705.
- [NIPS]Jonas Geiping, Hartmut Bauermeister, Hannah Dröge, Michael Moeller .[Inverting Gradients -- How easy is it to break privacy in federated learning?](https://arxiv.org/pdf/2003.14053) [J]. arXiv preprint arXiv:2003.14053.<br>[code:[JonasGeiping/invertinggradients](https://github.com/JonasGeiping/invertinggradients)]
- David Enthoven, Zaid Al-Ars .[An Overview of Federated Deep Learning Privacy Attacks and Defensive Strategies](https://arxiv.org/pdf/2004.04676) [J]. arXiv preprint arXiv:2004.04676.
- Amit Portnoy, Danny Hendler .[Towards Realistic Byzantine-Robust Federated Learning](https://arxiv.org/pdf/2004.04986) [J]. arXiv preprint arXiv:2004.04986.
- Gan Sun, Yang Cong (Senior Member, IEEE), Jiahua Dong, Qiang Wang, Ji Liu .[Data Poisoning Attacks on Federated Machine Learning](https://arxiv.org/pdf/2004.10020) [J]. arXiv preprint arXiv:2004.10020.
- Wenqi Wei, Ling Liu, Margaret Loper, Ka-Ho Chow, Mehmet Emre Gursoy, Stacey Truex, Yanzhao Wu .[A Framework for Evaluating Gradient Leakage Attacks in Federated Learning](https://arxiv.org/pdf/2004.10397) [J]. arXiv preprint arXiv:2004.10397.
- Xinjian Luo, Xiangqi Zhu .[Exploiting Defenses against GAN-Based Feature Inference Attacks in Federated Learning](https://arxiv.org/pdf/2004.12571) [J]. arXiv preprint arXiv:2004.12571.
- Renuga Kanagavelu, Zengxiang Li, Juniarto Samsudin, Yechao Yang, Feng Yang, Rick Siow Mong Goh, Mervyn Cheah, Praewpiraya Wiwatphonthana, Khajonpong Akkarajitsakul, Shangguang Wangz .[Two-Phase Multi-Party Computation Enabled Privacy-Preserving Federated Learning](https://arxiv.org/pdf/2005.11901) [J]. arXiv preprint arXiv:2005.11901.
- Chien-Lun Chen, Leana Golubchik, Marco Paolieri .[Backdoor Attacks on Federated Meta-Learning](https://arxiv.org/pdf/2006.07026) [J]. arXiv preprint arXiv:2006.07026.
- Zeou Hu, Kiarash Shaloudegi, Guojun Zhang, Yaoliang Yu .[FedMGDA+: Federated Learning meets Multi-objective Optimization](https://arxiv.org/pdf/2006.11489) [J]. arXiv preprint arXiv:2006.11489.
- Data D, Diggavi S. [Byzantine-Resilient High-Dimensional SGD with Local Iterations on Heterogeneous Data](https://arxiv.org/pdf/2006.13041)[J]. arXiv preprint arXiv:2006.13041, 2020.
- Song Y, Liu T, Wei T, et al. [FDA3: Federated Defense Against Adversarial Attacks for Cloud-Based IIoT Applications](https://arxiv.org/pdf/2006.15632)[J]. IEEE Transactions on Industrial Informatics, 2020.


## Privacy && Homomorphic Encryption
- [Baseline]Brendan McMahan H, Moore E, Ramage D, et al. [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)[J]. arXiv, 2016: arXiv: 1602.05629.
- [NIPS]Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H. Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, Karn Seth .[Practical Secure Aggregation for Federated Learning on User-Held Data](https://arxiv.org/pdf/1611.04482) [J]. arXiv preprint arXiv:1611.04482.
- Bonawitz K, Ivanov V, Kreuter B, et al. [Practical secure aggregation for privacy-preserving machine learning](https://www.researchgate.net/profile/Keith_Bonawitz/publication/320678967_Practical_Secure_Aggregation_for_Privacy-Preserving_Machine_Learning/links/5acb89dcaca272abdc635fc5/Practical-Secure-Aggregation-for-Privacy-Preserving-Machine-Learning.pdf)[C]//Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security. 2017: 1175-1191.
- Stephen Hardy, Wilko Henecka, Hamish Ivey-Law, Richard Nock, Giorgio Patrini, Guillaume Smith, Brian Thorne .[Private federated learning on vertically partitioned data via entity  resolution and additively homomorphic encryption](https://arxiv.org/pdf/1711.10677) [J]. arXiv preprint arXiv:1711.10677.
- McMahan H B, Ramage D, Talwar K, et al. [Learning differentially private recurrent language models](https://arxiv.org/pdf/1710.06963)[J]. arXiv preprint arXiv:1710.06963, 2017.
- [good]Robin C. Geyer, Tassilo Klein, Moin Nabi .[Differentially Private Federated Learning: A Client Level Perspective](https://arxiv.org/pdf/1712.07557) [J]. arXiv preprint arXiv:1712.07557.
- Papernot N, Song S, Mironov I, et al. [Scalable private learning with pate](https://arxiv.org/pdf/1802.08908.pdf?ref=hackernoon.com)[J]. arXiv preprint arXiv:1802.08908, 2018.
- Orekondy T, Oh S J, Zhang Y, et al. [Gradient-Leaks: Understanding and Controlling Deanonymization in Federated Learning](https://arxiv.org/pdf/1805.05838)[J]. arXiv preprint arXiv:1805.05838, 2018.
- [good]Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, Vitaly Shmatikov .[How To Backdoor Federated Learning](https://arxiv.org/pdf/1807.00459) [J]. arXiv preprint arXiv:1807.00459.<br>[code:[ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)]
- Ryffel T, Trask A, Dahl M, et al. [A generic framework for privacy preserving deep learning](https://arxiv.org/pdf/1811.04017.pdf!)[J]. arXiv preprint arXiv:1811.04017, 2018.
- [ICML]Arjun Nitin Bhagoji, Supriyo Chakraborty, Prateek Mittal, Seraphin Calo .[Analyzing Federated Learning through an Adversarial Lens](https://arxiv.org/pdf/1811.12470) [J]. arXiv preprint arXiv:1811.12470.<br>[code:[inspire-group/ModelPoisoning](https://github.com/inspire-group/ModelPoisoning)]
- Zhibo Wang, Mengkai Song, Zhifei Zhang, Yang Song, Qian Wang, Hairong Qi .[Beyond Inferring Class Representatives: User-Level Privacy Leakage From Federated Learning](https://arxiv.org/pdf/1812.00535) [J]. arXiv preprint arXiv:1812.00535.
- Vepakomma P, Gupta O, Dubey A, et al. [Reducing leakage in distributed deep learning for sensitive health data](https://www.researchgate.net/profile/Praneeth_Vepakomma2/publication/333171913_Reducing_leakage_in_distributed_deep_learning_for_sensitive_health_data/links/5cdeb29a299bf14d95a1772c/Reducing-leakage-in-distributed-deep-learning-for-sensitive-health-data.pdf)[J]. arXiv preprint arXiv:1812.00564, 2019.
- Milad Nasr, Reza Shokri, Amir Houmansadr .[Comprehensive Privacy Analysis of Deep Learning: Stand-alone and Federated Learning under Passive and Active White-box Inference Attacks](https://arxiv.org/pdf/1812.00910) [J]. arXiv preprint arXiv:1812.00910.
- Stacey Truex, Nathalie Baracaldo, Ali Anwar, Thomas Steinke, Heiko Ludwig, Rui Zhang .[A Hybrid Approach to Privacy-Preserving Federated Learning](https://arxiv.org/pdf/1812.03224) [J]. arXiv preprint arXiv:1812.03224.
- Zhao L, Wang Q, Zou Q, et al. [Privacy-preserving collaborative deep learning with unreliable participants](https://arxiv.org/pdf/1812.10113)[J]. IEEE Transactions on Information Forensics and Security, 2019, 15: 1486-1500.
- Aleksei Triastcyn, Boi Faltings .[Federated Generative Privacy](https://arxiv.org/pdf/1910.08385) [J]. arXiv preprint arXiv:1910.08385.
- Zhu L, Liu Z, Han S. [Deep leakage from gradients](https://papers.nips.cc/paper/9617-deep-leakage-from-gradients.pdf)[C]//Advances in Neural Information Processing Systems. 2019: 14774-14784.
- Kang Wei, Jun Li, Ming Ding, Chuan Ma, Howard H. Yang, Farokhi Farhad, Shi Jin, Tony Q. S. Quek, H. Vincent Poor .[Federated Learning with Differential Privacy: Algorithms and Performance Analysis](https://arxiv.org/pdf/1911.00222) [J]. arXiv preprint arXiv:1911.00222.
- Zaoxing Liu, Tian Li, Virginia Smith, Vyas Sekar .[Enhancing the Privacy of Federated Learning with Sketching](https://arxiv.org/pdf/1911.01812) [J]. arXiv preprint arXiv:1911.01812.
- Aleksei Triastcyn, Boi Faltings .[Federated Learning with Bayesian Differential Privacy](https://arxiv.org/pdf/1911.10071) [J]. arXiv preprint arXiv:1911.10071.
- Runhua Xu, Nathalie Baracaldo, Yi Zhou, Ali Anwar, Heiko Ludwig .[HybridAlpha: An Efficient Approach for Privacy-Preserving Federated Learning](https://arxiv.org/pdf/1912.05897) [J]. arXiv preprint arXiv:1912.05897.
- Daniel Peterson, Pallika Kanani, Virendra J. Marathe .[Private Federated Learning with Domain Adaptation](https://arxiv.org/pdf/1912.06733) [J]. arXiv preprint arXiv:1912.06733.
- Zhao B, Mopuri K R, Bilen H. [iDLG: Improved Deep Leakage from Gradients](https://arxiv.org/pdf/2001.02610)[J]. arXiv preprint arXiv:2001.02610, 2020
- Chen D, Orekondy T, Fritz M. [Gs-wgan: A gradient-sanitized approach for learning differentially private generators](https://proceedings.neurips.cc/paper/2020/file/9547ad6b651e2087bac67651aa92cd0d-Paper.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.
- Jeon J, Kim J, Kim J, et al. [Privacy-preserving deep learning computation for geo-distributed medical big-data platforms](https://arxiv.org/pdf/2001.02932)[C]//2019 49th Annual IEEE/IFIP International Conference on Dependable Systems and Networks–Supplemental Volume (DSN-S). IEEE, 2019: 3-4.
- Olivia Choudhury, Aris Gkoulalas-Divanis, Theodoros Salonidis, Issa Sylla, Yoonyoung Park, Grace Hsu, Amar Das .[Anonymizing Data for Privacy-Preserving Federated Learning](https://arxiv.org/pdf/2002.09096) [J]. arXiv preprint arXiv:2002.09096.
- Yan Feng, Xue Yang, Weijun Fang, Shu-Tao Xia, Xiaohu Tang .[Practical and Bilateral Privacy-preserving Federated Learning](https://arxiv.org/pdf/2002.09843) [J]. arXiv preprint arXiv:2002.09843.
- Ruixuan Liu, Yang Cao, Masatoshi Yoshikawa, Hong Chen .[FedSel: Federated SGD under Local Differential Privacy with Top-k Dimension Selection](https://arxiv.org/pdf/2003.10637) [J]. arXiv preprint arXiv:2003.10637.
- Katevas K, Bagdasaryan E, Waterman J, et al. [Decentralized Policy-Based Private Analytics](https://arxiv.org/pdf/2003.06612)[J]. arXiv preprint arXiv:2003.06612, 2020. 
- Yang Liu, Zhuo Ma, Ximeng Liu, Jianfeng Ma .[Learn to Forget: User-Level Memorization Elimination in Federated Learning](https://arxiv.org/pdf/2003.10933) [J]. arXiv preprint arXiv:2003.10933.
- Kalikinkar Mandal, Guang Gong .[PrivFL: Practical Privacy-preserving Federated Regressions on High-dimensional Data over Mobile Networks](https://arxiv.org/pdf/2004.02264) [J]. arXiv preprint arXiv:2004.02264.
- Yusuke Koda, Koji Yamamoto, Takayuki Nishio, Masahiro Morikura .[Differentially Private AirComp Federated Learning with Power Adaptation Harnessing Receiver Noise](https://arxiv.org/pdf/2004.06337) [J]. arXiv preprint arXiv:2004.06337.
- Fay D, Sjölund J, Oechtering T J. [Decentralized Differentially Private Segmentation with PATE](https://arxiv.org/pdf/2004.06567)[J]. arXiv preprint arXiv:2004.06567, 2020.
- Yang Zhao, Jun Zhao, Mengmeng Yang, Teng Wang, Ning Wang, Lingjuan Lyu, Dusit Niyato, Kwok Yan Lam .[Local Differential Privacy based Federated Learning for Internet of Things](https://arxiv.org/pdf/2004.08856) [J]. arXiv preprint arXiv:2004.08856.
- Aidmar Wainakh, Alejandro Sanchez Guinea, Tim Grube, Max Mühlhäuser .[Enhancing Privacy via Hierarchical Federated Learning](https://arxiv.org/pdf/2004.11361) [J]. arXiv preprint arXiv:2004.11361.
- M.A.P. Chamikara, P.Bertok, I.Khalil, D.Liu, S.Camtepe .[Privacy Preserving Distributed Machine Learning with Federated Learning](https://arxiv.org/pdf/2004.12108) [J]. arXiv preprint arXiv:2004.12108.
- Zhicong Liang, Bao Wang, Quanquan Gu, Stanley Osher, Yuan Yao .[Exploring Private Federated Learning with Laplacian Smoothing](https://arxiv.org/pdf/2005.00218) [J]. arXiv preprint arXiv:2005.00218.
- Semih Yagli, Alex Dytso, H. Vincent Poor .[Information-Theoretic Bounds on the Generalization Error and Privacy Leakage in Federated Learning](https://arxiv.org/pdf/2005.02503) [J]. arXiv preprint arXiv:2005.02503.
- Fagbohungbe O, Reza S R, Dong X, et al. [Efficient Privacy Preserving Edge Computing Framework for Image Classification](https://arxiv.org/pdf/2005.04563)[J]. arXiv preprint arXiv:2005.04563, 2020.
- Will Abramson, Adam James Hall, Pavlos Papadopoulos, Nikolaos Pitropakis, William J Buchanan. [A Distributed Trust Framework for Privacy-Preserving Machine Learning](https://arxiv.org/abs/2006.02456)[J]. arXiv preprint arXiv:2006.02456
- Stacey Truex, Ling Liu, Ka-Ho Chow, Mehmet Emre Gursoy, Wenqi Wei .[LDP-Fed: Federated Learning with Local Differential Privacy](https://arxiv.org/pdf/2006.03637) [J]. arXiv preprint arXiv:2006.03637.
- Lie He, Sai Praneeth Karimireddy, Martin Jaggi. [Secure Byzantine-Robust Machine Learning](https://arxiv.org/abs/2006.04747)[J]. arXiv preprint arXiv:2006.04747 
- Dongzhu Liu, Osvaldo Simeone .[Privacy For Free: Wireless Federated Learning Via Uncoded Transmission With Adaptive Power Control](https://arxiv.org/pdf/2006.05459) [J]. arXiv preprint arXiv:2006.05459.
- César Sabater, Aurélien Bellet, Jan Ramon. [Distributed Differentially Private Averaging with Improved Utility and Robustness to Malicious Parties](https://arxiv.org/abs/2006.07218)[J]. arXiv preprint arXiv:2006.07218


## Incentive Mechanism && Fairness
- Zhan Y, Li P, Qu Z, et al. [A learning-based incentive mechanism for federated learning](https://www.u-aizu.ac.jp/~pengli/files/fl_incentive_iot.pdf)[J]. IEEE Internet of Things Journal, 2020.
- [ICML]Mehryar Mohri, Gary Sivek, Ananda Theertha Suresh .[Agnostic Federated Learning](https://arxiv.org/pdf/1902.00146) [J]. arXiv preprint arXiv:1902.00146.
- [good]Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konecny, Stefano Mazzocchi, H. Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, Jason Roselander .[Towards Federated Learning at Scale: System Design](https://arxiv.org/pdf/1902.01046) [J]. arXiv preprint arXiv:1902.01046.
- Jiawen Kang, Zehui Xiong, Dusit Niyato, Han Yu, Ying-Chang Liang, Dong In Kim .[Incentive Design for Efficient Federated Learning in Mobile Networks: A Contract Theory Approach](https://arxiv.org/pdf/1905.07479) [J]. arXiv preprint arXiv:1905.07479.
- [ICLR]Tian Li, Maziar Sanjabi, Virginia Smith .[Fair Resource Allocation in Federated Learning](https://arxiv.org/pdf/1905.10497) [J]. arXiv preprint arXiv:1905.10497.<br>[code:[litian96/fair_flearn](https://github.com/litian96/fair_flearn)]
- Lyu L, Yu J, Nandakumar K, et al. [Towards Fair and Privacy-Preserving Federated Deep Models](https://arxiv.org/pdf/1906.01167)[J]. IEEE Transactions on Parallel and Distributed Systems, 2020, 31(11): 2524-2541.
- Yunus Sarikaya, Ozgur Ercetin .[Motivating Workers in Federated Learning: A Stackelberg Game Perspective](https://arxiv.org/pdf/1908.03092) [J]. arXiv preprint arXiv:1908.03092.
- Latif U. Khan, Nguyen H. Tran, Shashi Raj Pandey, Walid Saad, Zhu Han, Minh N. H. Nguyen, Choong Seon Hong .[Federated Learning for Edge Networks: Resource Optimization and Incentive Mechanism](https://arxiv.org/pdf/1911.05642) [J]. arXiv preprint arXiv:1911.05642.
- Yutao Jiao, Ping Wang, Dusit Niyato, Bin Lin, Dong In Kim .[Toward an Automated Auction Framework for Wireless Federated Learning Services Market](https://arxiv.org/pdf/1912.06370) [J]. arXiv preprint arXiv:1912.06370.
- Rongfei Zeng, Shixun Zhang, Jiaqi Wang, Xiaowen Chu .[FMore: An Incentive Scheme of Multi-dimensional Auction for Federated Learning in MEC](https://arxiv.org/pdf/2002.09699) [J]. arXiv preprint arXiv:2002.09699.
- Jingfeng Zhang, Cheng Li, Antonio Robles-Kelly, Mohan Kankanhalli .[Hierarchically Fair Federated Learning](https://arxiv.org/pdf/2004.10386) [J]. arXiv preprint arXiv:2004.10386.


## Communication-Efficiency
- [good]Jakub Konečný, H. Brendan McMahan, Felix X. Yu, Peter Richtárik, Ananda Theertha Suresh, Dave Bacon .[Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/pdf/1610.05492) [J]. arXiv preprint arXiv:1610.05492.
- Yujun Lin, Song Han, Huizi Mao, Yu Wang, William J. Dally. [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)[J]. arXiv preprint arXiv:1712.01887
- Nilsson A, Smith S, Ulm G, et al. [A performance evaluation of federated learning algorithms](https://www.researchgate.net/profile/Gregor_Ulm/publication/329106719_A_Performance_Evaluation_of_Federated_Learning_Algorithms/links/5c0fabcfa6fdcc494febf907/A-Performance-Evaluation-of-Federated-Learning-Algorithms.pdf)[C]//Proceedings of the Second Workshop on Distributed Infrastructures for Deep Learning. 2018: 1-8.
- Bui T D, Nguyen C V, Swaroop S, et al. [Partitioned variational inference: A unified framework encompassing federated and continual learning](https://arxiv.org/pdf/1811.11206)[J]. arXiv preprint arXiv:1811.11206, 2018.
- [good]Sebastian Caldas, Jakub Konečny, H. Brendan McMahan, Ameet Talwalkar .[Expanding the Reach of Federated Learning by Reducing Client Resource Requirements](https://arxiv.org/pdf/1812.07210) [J]. arXiv preprint arXiv:1812.07210.
- Hangyu Zhu, Yaochu Jin .[Multi-objective Evolutionary Federated Learning](https://arxiv.org/pdf/1812.07478) [J]. arXiv preprint arXiv:1812.07478.
- Neel Guha, Ameet Talwalkar, Virginia Smith .[One-Shot Federated Learning](https://arxiv.org/pdf/1902.11175) [J]. arXiv preprint arXiv:1902.11175.
- Yang Chen, Xiaoyan Sun, Yaochu Jin .[Communication-Efficient Federated Deep Learning with Asynchronous Model Update and Temporally Weighted Aggregation](https://arxiv.org/pdf/1903.07424) [J]. arXiv preprint arXiv:1903.07424.
- Chenghao Hu, Jingyan Jiang, Zhi Wang .[Decentralized Federated Learning: A Segmented Gossip Approach](https://arxiv.org/pdf/1908.07782) [J]. arXiv preprint arXiv:1908.07782.
- Abhishek Singh, Praneeth Vepakomma, Otkrist Gupta, Ramesh Raskar .[Detailed comparison of communication efficiency of split learning and federated learning](https://arxiv.org/pdf/1909.09145) [J]. arXiv preprint arXiv:1909.09145.
- Wentai Wu, Ligang He, Weiwei Lin, RuiMao, Stephen Jarvis .[SAFA: a Semi-Asynchronous Protocol for Fast Federated Learning with Low Overhead](https://arxiv.org/pdf/1910.01355) [J]. arXiv preprint arXiv:1910.01355.
- Yuqing Du, Sheng Yang, Kaibin Huang. [High-Dimensional Stochastic Gradient Quantization for Communication-Efficient Edge Learning](https://arxiv.org/abs/1910.03865)[J]. arXiv preprint arXiv:1019.03865
- Yan Z. [Gradient Sparification for Asynchronous Distributed Training](https://arxiv.org/pdf/1910.10929)[J]. arXiv preprint arXiv:1910.10929, 2019. 
- Anis Elgabli, Jihong Park, Sabbir Ahmed, Mehdi Bennis .[L-FGADMM: Layer-Wise Federated Group ADMM for Communication Efficient Decentralized Deep Learning](https://arxiv.org/pdf/1911.03654) [J]. arXiv preprint arXiv:1911.03654.
- Xinyan Dai, Xiao Yan, Kaiwen Zhou, Han Yang, Kelvin K. W. Ng, James Cheng, Yu Fan .[Hyper-Sphere Quantization: Communication-Efficient SGD for Federated Learning](https://arxiv.org/pdf/1911.04655) [J]. arXiv preprint arXiv:1911.04655.
- Asad M, Moustafa A, Ito T. [FedOpt: Towards Communication Efficiency and Privacy Preservation in Federated Learning](https://www.mdpi.com/2076-3417/10/8/2864/pdf)[J]. Applied Sciences, 2020, 10(8): 2864.
- Haozhao Wang, Zhihao Qu, Song Guo, Xin Gao, Ruixuan Li, Baoliu Ye .[Intermittent Pulling with Local Compensation for Communication-Efficient Federated Learning](https://arxiv.org/pdf/2001.08277) [J]. arXiv preprint arXiv:2001.08277.
- Anbu Huang, Yuanyuan Chen, Yang Liu, Tianjian Chen, Qiang Yang .[RPN: A Residual Pooling Network for Efficient Federated Learning](https://arxiv.org/pdf/2001.08600) [J]. arXiv preprint arXiv:2001.08600.
- Tang Z, Shi S, Chu X. [Communication-efficient decentralized learning with sparsification and adaptive peer selection](https://arxiv.org/pdf/2002.09692)[J]. arXiv preprint arXiv:2002.09692, 2020.
- Naifu Zhang, Meixia Tao .[Gradient Statistics Aware Power Control for Over-the-Air Federated Learning in Fading Channels](https://arxiv.org/pdf/2003.02089) [J]. arXiv preprint arXiv:2003.02089.
- Jinjin Xu, Wenli Du, Ran Cheng, Wangli He, Yaochu Jin .[Ternary Compression for Communication-Efficient Federated Learning](https://arxiv.org/pdf/2003.03564) [J]. arXiv preprint arXiv:2003.03564.
- Shaoxiong Ji, Wenqi Jiang, Anwar Walid, Xue Li .[Dynamic Sampling and Selective Masking for Communication-Efficient Federated Learning](https://arxiv.org/pdf/2003.09603) [J]. arXiv preprint arXiv:2003.09603.
- Muhammad Asad, Ahmed Moustafa, Takayuki Ito, Muhammad Aslam .[Evaluating the Communication Efficiency in Federated Learning Algorithms](https://arxiv.org/pdf/2004.02738) [J]. arXiv preprint arXiv:2004.02738.
- Mohammad Mohammadi Amiri, Deniz Gunduz, Sanjeev R. Kulkarni, H. Vincent Poor .[Federated Learning With Quantized Global Model Updates](https://arxiv.org/pdf/2006.10672) [J]. arXiv preprint arXiv:2006.10672.
- Horváth S, Richtárik P. [A Better Alternative to Error Feedback for Communication-Efficient Distributed Learning](https://arxiv.org/pdf/2006.11077)[J]. arXiv preprint arXiv:2006.11077, 2020.
- Xiang Ma, Haijian Sun, Rose Qingyang Hu .[Scheduling Policy and Power Allocation for Federated Learning in NOMA Based MEC](https://arxiv.org/pdf/2006.13044) [J]. arXiv preprint arXiv:2006.13044.
- Constantin Philippenko, Aymeric Dieuleveut .[Artemis: tight convergence guarantees for bidirectional compression in Federated Learning](https://arxiv.org/pdf/2006.14591) [J]. arXiv preprint arXiv:2006.14591.
- Shen T, Zhang J, Jia X, et al. [Federated Mutual Learning](https://arxiv.org/pdf/2006.16765)[J]. arXiv preprint arXiv:2006.16765, 2020.
- Farzin Haddadpour, Mohammad Mahdi Kamani, Aryan Mokhtari, Mehrdad Mahdavi .[Federated Learning with Compression: Unified Analysis and Sharp Guarantees](https://arxiv.org/pdf/2007.01154) [J]. arXiv preprint arXiv:2007.01154.


## Straggler Problem
- Sukjong Ha, Jingjing Zhang, Osvaldo Simeone, Joonhyuk Kang .[Coded Federated Computing in Wireless Networks with Straggling Devices and Imperfect CSI](https://arxiv.org/pdf/1901.05239) [J]. arXiv preprint arXiv:1901.05239.
- Linara Adilova, Julia Rosenzweig, Michael Kamp .[Information-Theoretic Perspective of Federated Learning](https://arxiv.org/pdf/1911.07652) [J]. arXiv preprint arXiv:1911.07652.
- Jinhyun So, Basak Guler, A. Salman Avestimehr .[Turbo-Aggregate: Breaking the Quadratic Aggregation Barrier in Secure Federated Learning](https://arxiv.org/pdf/2002.04156) [J]. arXiv preprint arXiv:2002.04156.
- Sagar Dhakal, Saurav Prakash, Yair Yona, Shilpa Talwar, Nageen Himayat .[Coded Federated Learning](https://arxiv.org/pdf/2002.09574) [J]. arXiv preprint arXiv:2002.09574.


## Computation Efficiency
- Li L, Xiong H, Guo Z, et al. [SmartPC: Hierarchical Pace Control in Real-Time Federated Learning System](https://www.ece.ucf.edu/~zsguo/pubs/conference_workshop/RTSS2019b.pdf)[C]//2019 IEEE Real-Time Systems Symposium (RTSS). IEEE, 2019: 406-418.
- Kumar D, Ramkumar A A, Sindhu R, et al. [Decaf: Iterative collaborative processing over the edge](https://www.usenix.org/system/files/hotedge19-paper-kumar.pdf)[C]//2nd {USENIX} Workshop on Hot Topics in Edge Computing (HotEdge 19). 2019.
- Vepakomma, Praneeth, et al. [Split Learning for Health: Distributed Deep Learning without Sharing Raw Patient Data.](https://arxiv.org/pdf/1812.00564) ArXiv Preprint ArXiv:1812.00564, 2018.
- Jinke Ren, Guanding Yu, Guangyao Ding .[Accelerating DNN Training in Wireless Federated Edge Learning System](https://arxiv.org/pdf/1905.09712) [J]. arXiv preprint arXiv:1905.09712.
- Vito Walter Anelli, Yashar Deldjoo, Tommaso Di Noia, Antonio Ferrara .[Towards Effective Device-Aware Federated Learning](https://arxiv.org/pdf/1908.07420) [J]. arXiv preprint arXiv:1908.07420.
- Yuang Jiang, Shiqiang Wang, Bong Jun Ko, Wei-Han Lee, Leandros Tassiulas .[Model Pruning Enables Efficient Federated Learning on Edge Devices](https://arxiv.org/pdf/1909.12326) [J]. arXiv preprint arXiv:1909.12326.
- Nicolas Skatchkovsky, Hyeryung Jang, Osvaldo Simeone .[Federated Neuromorphic Learning of Spiking Neural Networks for Low-Power Edge Intelligence](https://arxiv.org/pdf/1910.09594) [J]. arXiv preprint arXiv:1910.09594.
- Yujing Chen, Yue Ning, Huzefa Rangwala .[Asynchronous Online Federated Learning for Edge Devices](https://arxiv.org/pdf/1911.02134) [J]. arXiv preprint arXiv:1911.02134.
- Chaoyue Niu, Fan Wu, Shaojie Tang, Lifeng Hua, Rongfei Jia, Chengfei Lv, Zhihua Wu, Guihai Chen .[Secure Federated Submodel Learning](https://arxiv.org/pdf/1911.02254) [J]. arXiv preprint arXiv:1911.02254.
- Zirui Xu, Zhao Yang, Jinjun Xiong, Jianlei Yang, Xiang Chen .[ELFISH: Resource-Aware Federated Learning on Heterogeneous Edge Devices](https://arxiv.org/pdf/1912.01684) [J]. arXiv preprint arXiv:1912.01684.
- Martin Isaksson, Karl Norrman .[Secure Federated Learning in 5G Mobile Networks](https://arxiv.org/pdf/2004.06700) [J]. arXiv preprint arXiv:2004.06700.
- Sohei Itahara, Takayuki Nishio, Masahiro Morikura, Koji Yamamoto .[Lottery Hypothesis based Unsupervised Pre-training for Model Compression in Federated Learning](https://arxiv.org/pdf/2004.09817) [J]. arXiv preprint arXiv:2004.09817.
- Chandra Thapa, M.A.P. Chamikara, Seyit Camtepe .[SplitFed: When Federated Learning Meets Split Learning](https://arxiv.org/pdf/2004.12088) [J]. arXiv preprint arXiv:2004.12088.
- Rapp M, Khalili R, Henkel J. [Distributed Learning on Heterogeneous Resource-Constrained Devices](https://arxiv.org/pdf/2006.05403)[J]. arXiv preprint arXiv:2006.05403, 2020.


## Wireless Communication && Cloud Computing && networking
- Feraudo A, Yadav P, Safronov V, et al. [CoLearn: enabling federated learning in MUD-compliant IoT edge networks](https://www.researchgate.net/profile/Poonam_Yadav14/publication/341424819_CoLearn_Enabling_Federated_Learning_in_MUD-compliant_IoT_Edge_Networks/links/5ebf7fc5299bf1c09ac0b5dd/CoLearn-Enabling-Federated-Learning-in-MUD-compliant-IoT-Edge-Networks.pdf)[C]//Proceedings of the Third ACM International Workshop on Edge Systems, Analytics and Networking. 2020: 25-30.
- Umair Mohammad, Sameh Sorour .[Adaptive Task Allocation for Asynchronous Federated Mobile Edge Learning](https://arxiv.org/pdf/1905.01656) [J]. arXiv preprint arXiv:1905.01656.
- Xiaofei Wang, Yiwen Han, Chenyang Wang, Qiyang Zhao, Xu Chen, Min Chen .[In-Edge AI: Intelligentizing Mobile Edge Computing, Caching and  Communication by Federated Learning](https://arxiv.org/pdf/1809.07857) [J]. arXiv preprint arXiv:1809.07857.
- Shaohan Feng, Dusit Niyato, Ping Wang, Dong In Kim, Ying-Chang Liang .[Joint Service Pricing and Cooperative Relay Communication for Federated Learning](https://arxiv.org/pdf/1811.12082) [J]. arXiv preprint arXiv:1811.12082.
- Mingzhe Chen, Omid Semiari, Walid Saad, Xuanlin Liu, Changchuan Yin .[Federated Echo State Learning for Minimizing Breaks in Presence in Wireless Virtual Reality Networks](https://arxiv.org/pdf/1812.01202) [J]. arXiv preprint arXiv:1812.01202.
- Guangxu Zhu, Yong Wang, Kaibin Huang .[Low-Latency Broadband Analog Aggregation for Federated Edge Learning](https://arxiv.org/pdf/1812.11494) [J]. arXiv preprint arXiv:1812.11494.
- Kai Yang, Tao Jiang, Yuanming Shi, Zhi Ding .[Federated Learning via Over-the-Air Computation](https://arxiv.org/pdf/1812.11750) [J]. arXiv preprint arXiv:1812.11750.
- Tran N H, Bao W, Zomaya A, et al. [Federated learning over wireless networks: Optimization model design and analysis](http://163.180.116.116/layouts/net/publications/data/2019)Federated%20Learning%20over%20Wireless%20Network.pdf)[C]//IEEE INFOCOM 2019-IEEE Conference on Computer Communications. IEEE, 2019: 1387-1395.
- Amiri M M, Gündüz D. [Machine learning at the wireless edge: Distributed stochastic gradient descent over-the-air](https://arxiv.org/pdf/1901.00844)[J]. IEEE Transactions on Signal Processing, 2020, 68: 2155-2169.
- Guan Wang .[Interpret Federated Learning with Shapley Values](https://arxiv.org/pdf/1905.04519) [J]. arXiv preprint arXiv:1905.04519.
- Qian J, Sengupta S, Hansen L K. [Active learning solution on distributed edge computing](https://arxiv.org/pdf/1906.10718)[J]. arXiv preprint arXiv:1906.10718, 2019.
- Yang Zhao, Jun Zhao, Linshan Jiang, Rui Tan, Dusit Niyato .[Mobile Edge Computing, Blockchain and Reputation-based Crowdsourcing IoT Federated Learning: A Secure, Decentralized and Privacy-preserving System](https://arxiv.org/pdf/1906.10893) [J]. arXiv preprint arXiv:1906.10893.
- Qunsong Zeng, Yuqing Du, Kin K. Leung, Kaibin Huang .[Energy-Efficient Radio Resource Allocation for Federated Edge Learning](https://arxiv.org/pdf/1907.06040) [J]. arXiv preprint arXiv:1907.06040.
- Mohammad Mohammadi Amiri, Deniz Gunduz .[Federated Learning over Wireless Fading Channels](https://arxiv.org/pdf/1907.09769) [J]. arXiv preprint arXiv:1907.09769.
- Evita Bakopoulou, Balint Tillman, Athina Markopoulou .[A Federated Learning Approach for Mobile Packet Classification](https://arxiv.org/pdf/1907.13113) [J]. arXiv preprint arXiv:1907.13113.
- Xin Yao, Tianchi Huang, Chenglei Wu, Rui-Xiao Zhang, Lifeng Sun .[Federated Learning with Additional Mechanisms on Clients to Reduce Communication Costs](https://arxiv.org/pdf/1908.05891) [J]. arXiv preprint arXiv:1908.05891.
- Howard H. Yang, Zuozhu Liu, Tony Q. S. Quek, H. Vincent Poor .[Scheduling Policies for Federated Learning in Wireless Networks](https://arxiv.org/pdf/1908.06287) [J]. arXiv preprint arXiv:1908.06287.
- Mehdi Salehi Heydar Abad, Emre Ozfatura, Deniz Gunduz, Ozgur Ercetin .[Hierarchical Federated Learning Across Heterogeneous Cellular Networks](https://arxiv.org/pdf/1909.02362) [J]. arXiv preprint arXiv:1909.02362.
- Chuan Ma, Jun Li, Ming Ding, Howard Hao Yang, Feng Shu, Tony Q. S. Quek, H. Vincent Poor .[On Safeguarding Privacy and Security in the Framework of Federated Learning](https://arxiv.org/pdf/1909.06512) [J]. arXiv preprint arXiv:1909.06512.
- Mingzhe Chen, Zhaohui Yang, Walid Saad, Changchuan Yin, H. Vincent Poor, Shuguang Cui .[A Joint Learning and Communications Framework for Federated Learning over Wireless Networks](https://arxiv.org/pdf/1909.07972) [J]. arXiv preprint arXiv:1909.07972.
- Tung T. Vu, Duy T. Ngo, Nguyen H. Tran, Hien Quoc Ngo, Minh N. Dao, Richard H. Middleton .[Cell-Free Massive MIMO for Wireless Federated Learning](https://arxiv.org/pdf/1909.12567) [J]. arXiv preprint arXiv:1909.12567.
- Jack Goetz, Kshitiz Malik, Duc Bui, Seungwhan Moon, Honglei Liu, Anuj Kumar .[Active Federated Learning](https://arxiv.org/pdf/1909.12641) [J]. arXiv preprint arXiv:1909.12641.
- Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jadbabaie, Ramtin Pedarsani .[FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization](https://arxiv.org/pdf/1909.13014) [J]. arXiv preprint arXiv:1909.13014.
- Jiawen Kang, Zehui Xiong, Dusit Niyato, Yuze Zou, Yang Zhang, Mohsen Guizani .[Reliable Federated Learning for Mobile Networks](https://arxiv.org/pdf/1910.06837) [J]. arXiv preprint arXiv:1910.06837.
- Huy T. Nguyen, Nguyen Cong Luong, Jun Zhao, Chau Yuen, Dusit Niyato .[Resource Allocation in Mobility-Aware Federated Learning Networks: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/1910.09172) [J]. arXiv preprint arXiv:1910.09172.
- Canh Dinh, Nguyen H. Tran, Minh N. H. Nguyen, Choong Seon Hong, Wei Bao, Albert Y. Zomaya, Vincent Gramoli .[Federated Learning over Wireless Networks: Convergence Analysis and Resource Allocation](https://arxiv.org/pdf/1910.13067) [J]. arXiv preprint arXiv:1910.13067.
- Howard H. Yang, Ahmed Arafa, Tony Q. S. Quek, H. Vincent Poor .[Age-Based Scheduling Policy for Federated Learning in Mobile Edge Networks](https://arxiv.org/pdf/1910.14648) [J]. arXiv preprint arXiv:1910.14648.
- Yuxuan Sun, Sheng Zhou, Deniz Gündüz .[Energy-Aware Analog Aggregation for Federated Learning with Redundant Data](https://arxiv.org/pdf/1911.00188) [J]. arXiv preprint arXiv:1911.00188.
- Wenqi Shi, Sheng Zhou, Zhisheng Niu .[Device Scheduling with Fast Convergence for Wireless Federated Learning](https://arxiv.org/pdf/1911.00856) [J]. arXiv preprint arXiv:1911.00856.
- Zhaohui Yang, Mingzhe Chen, Walid Saad, Choong Seon Hong, Mohammad Shikh-Bahaei .[Energy Efficient Federated Learning Over Wireless Communication Networks](https://arxiv.org/pdf/1911.02417) [J]. arXiv preprint arXiv:1911.02417.
- Jun Li, Xiaoman Shen, Lei Chen, Jiajia Chen .[Bandwidth Slicing to Boost Federated Learning in Edge Computing](https://arxiv.org/pdf/1911.07615) [J]. arXiv preprint arXiv:1911.07615.
- Keith Bonawitz, Fariborz Salehi, Jakub Konečný, Brendan McMahan, Marco Gruteser .[Federated Learning with Autotuned Communication-Efficient Secure Aggregation](https://arxiv.org/pdf/1912.00131) [J]. arXiv preprint arXiv:1912.00131.
- Jinho Choi, Shiva Raj Pokhrel .[Federated learning with multichannel ALOHA](https://arxiv.org/pdf/1912.06273) [J]. arXiv preprint arXiv:1912.06273.
- Yanan Li, Shusen Yang, Xuebin Ren, Cong Zhao .[Asynchronous Federated Learning with Differential Privacy for Edge Intelligence](https://arxiv.org/pdf/1912.07902) [J]. arXiv preprint arXiv:1912.07902.
- Stefano Savazzi, Monica Nicoli, Vittorio Rampa .[Federated Learning with Cooperating Devices: A Consensus Approach for Massive IoT Networks](https://arxiv.org/pdf/1912.13163) [J]. arXiv preprint arXiv:1912.13163.
- Guangxu Zhu, Yuqing Du, Deniz Gunduz, Kaibin Huang .[One-Bit Over-the-Air Aggregation for Communication-Efficient Federated Edge Learning: Design and Convergence Analysis](https://arxiv.org/pdf/2001.05713) [J]. arXiv preprint arXiv:2001.05713.
- Mingzhe Chen, H. Vincent Poor, Walid Saad, Shuguang Cui .[Convergence Time Optimization for Federated Learning over Wireless Networks](https://arxiv.org/pdf/2001.07845) [J]. arXiv preprint arXiv:2001.07845.
- Wei-Ting Chang, Ravi Tandon .[Communication Efficient Federated Learning over Multiple Access Channels](https://arxiv.org/pdf/2001.08737) [J]. arXiv preprint arXiv:2001.08737.
- Mohammad Mohammadi Amiri, Deniz Gunduz, Sanjeev R. Kulkarni, H. Vincent Poor .[Update Aware Device Scheduling for Federated Learning at the Wireless Edge](https://arxiv.org/pdf/2001.10402) [J]. arXiv preprint arXiv:2001.10402.
- Chakraborty S, Mohammed H, Saha D. [Learning from Peers at the Wireless Edge](https://arxiv.org/pdf/2001.11567.pdf)[C]//2020 International Conference on COMmunication Systems & NETworkS (COMSNETS). IEEE, 2020: 779-784.
- Mohamed Seif, Ravi Tandon, Ming Li .[Wireless Federated Learning with Local Differential Privacy](https://arxiv.org/pdf/2002.05151) [J]. arXiv preprint arXiv:2002.05151.
- Tengchan Zeng, Omid Semiari, Mohammad Mozaffari, Mingzhe Chen, Walid Saad, Mehdi Bennis .[Federated Learning in the Sky: Joint Power Allocation and Scheduling with UAV Swarms](https://arxiv.org/pdf/2002.08196) [J]. arXiv preprint arXiv:2002.08196.
- Hong Xing, Osvaldo Simeone, Suzhi Bi .[Decentralized Federated Learning via SGD over Wireless D2D Networks](https://arxiv.org/pdf/2002.12507) [J]. arXiv preprint arXiv:2002.12507.
- Praneeth Narayanamurthy, Namrata Vaswani, Aditya Ramamoorthy .[Federated Over-the-Air Subspace Learning from Incomplete Data](https://arxiv.org/pdf/2002.12873) [J]. arXiv preprint arXiv:2002.12873.
- Xiaopeng Mo, Jie Xu .[Energy-Efficient Federated Edge Learning with Joint Communication and Computation Design](https://arxiv.org/pdf/2003.00199) [J]. arXiv preprint arXiv:2003.00199.
- Kang Wei, Jun Li, Ming Ding, Chuan Ma, Hang Su, Bo Zhang, H. Vincent Poor .[Performance Analysis and Optimization in Privacy-Preserving Federated Learning](https://arxiv.org/pdf/2003.00229) [J]. arXiv preprint arXiv:2003.00229.
- Haijian Sun, Xiang Ma, Rose Qingyang Hu .[Adaptive Federated Learning With Gradient Compression in Uplink NOMA](https://arxiv.org/pdf/2003.01344) [J]. arXiv preprint arXiv:2003.01344.
- Yo-Seb Jeon, Mohammad Mohammadi Amiri, Jun Li, H. Vincent Poor .[Gradient Estimation for Federated Learning over Massive MIMO Communication Systems](https://arxiv.org/pdf/2003.08059) [J]. arXiv preprint arXiv:2003.08059.
- Sihua Wang, Mingzhe Chen, Changchuan Yin, Walid Saad, Choong Seon Hong, Shuguang Cui, H. Vincent Poor .[Federated Learning for Task and Resource Allocation in Wireless High Altitude Balloon Networks](https://arxiv.org/pdf/2003.09375) [J]. arXiv preprint arXiv:2003.09375.
- Rui Hu, Yuanxiong Guo, E. Paul. Ratazzi, Yanmin Gong .[Differentially Private Federated Learning for Resource-Constrained Internet of Things](https://arxiv.org/pdf/2003.12705) [J]. arXiv preprint arXiv:2003.12705.
- Jinke Ren, Yinghui He, Dingzhu Wen, Guanding Yu, Kaibin Huang, Dongning Guo .[Scheduling in Cellular Federated Edge Learning with Importance and Channel Awareness](https://arxiv.org/pdf/2004.00490) [J]. arXiv preprint arXiv:2004.00490.
- Yuzheng Li, Chuan Chen, Nan Liu, Huawei Huang, Zibin Zheng, Qiang Yan .[A Blockchain-based Decentralized Federated Learning Framework with Committee Consensus](https://arxiv.org/pdf/2004.00773) [J]. arXiv preprint arXiv:2004.00773.
- Nguyen Quang Hieu, Tran The Anh, Nguyen Cong Luong, Dusit Niyato, Dong In Kim, Erik Elmroth .[Resource Management for Blockchain-enabled Federated Learning: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/2004.04104) [J]. arXiv preprint arXiv:2004.04104.
- Jie Xu, Heqiang Wang .[Client Selection and Bandwidth Allocation in Wireless Federated Learning Networks: A Long-Term Perspective](https://arxiv.org/pdf/2004.04314) [J]. arXiv preprint arXiv:2004.04314.
- Kai Yang, Yuanming Shi, Yong Zhou, Zhanpeng Yang, Liqun Fu, Wei Chen .[Federated Machine Learning for Intelligent IoT via Reconfigurable Intelligent Surface](https://arxiv.org/pdf/2004.05843) [J]. arXiv preprint arXiv:2004.05843.
- Martin Isaksson, Karl Norrman .[Secure Federated Learning in 5G Mobile Networks](https://arxiv.org/pdf/2004.06700) [J]. arXiv preprint arXiv:2004.06700.
- Richeng Jin, Xiaofan He, Huaiyu Dai .[On the Design of Communication Efficient Federated Learning over Wireless Networks](https://arxiv.org/pdf/2004.07351) [J]. arXiv preprint arXiv:2004.07351.
- Meng Jiang, Taeho Jung, Ryan Karl, Tong Zhao .[Federated Dynamic GNN with Secure Aggregation](https://arxiv.org/pdf/2009.07351) [J]. arXiv preprint arXiv:2009.07351.
- Tu Y, Ruan Y, Wang S, et al. [Network-Aware Optimization of Distributed Learning for Fog Computing](https://arxiv.org/pdf/2004.08488)[J]. arXiv preprint arXiv:2004.08488, 2020.
- Yu D, Park S H, Simeone O, et al. [Optimizing Over-the-Air Computation in IRS-Aided C-RAN Systems](https://arxiv.org/pdf/2004.09168)[J]. arXiv preprint arXiv:2004.09168, 2020.
- Yong Xiao, Guangming Shi, Marwan Krunz .[Towards Ubiquitous AI in 6G with Federated Learning](https://arxiv.org/pdf/2004.13563) [J]. arXiv preprint arXiv:2004.13563.
- Ha-Vu Tran, Georges Kaddoum, Hany Elgala, Chadi Abou-Rjeily, Hemani Kaushal .[Lightwave Power Transfer for Federated Learning-based Wireless Networks](https://arxiv.org/pdf/2005.03977) [J]. arXiv preprint arXiv:2005.03977.
- Zhijin Qin, Geoffrey Ye Li, Hao Ye .[Federated Learning and Wireless Communications](https://arxiv.org/pdf/2005.05265) [J]. arXiv preprint arXiv:2005.05265.
- Yi Liu, Jialiang Peng, Jiawen Kang, Abdullah M. Iliyasu, Dusit Niyato, Ahmed A. Abd El-Latif .[A Secure Federated Learning Framework for 5G Networks](https://arxiv.org/pdf/2005.05752) [J]. arXiv preprint arXiv:2005.05752.
- Amir Sonee, Stefano Rini .[Efficient Federated Learning over Multiple Access Channel with Differential Privacy Constraints](https://arxiv.org/pdf/2005.07776) [J]. arXiv preprint arXiv:2005.07776.
- Ahmet M. Elbir, Sinem Coleri .[Federated Deep Learning Framework For Hybrid Beamforming in mm-Wave Massive MIMO](https://arxiv.org/pdf/2005.09969) [J]. arXiv preprint arXiv:2005.09969.
- Mingzhe Chen, H. Vincent Poor, Walid Saad, Shuguang Cui .[Wireless Communications for Collaborative Federated Learning in the Internet of Things](https://arxiv.org/pdf/2006.02499) [J]. arXiv preprint arXiv:2006.02499.
- Nir Shlezinger, Mingzhe Chen, Yonina C. Eldar, H. Vincent Poor, Shuguang Cui .[UVeQFed: Universal Vector Quantization for Federated Learning](https://arxiv.org/pdf/2006.03262) [J]. arXiv preprint arXiv:2006.03262.
- Seungeun Oh, Jihong Park, Eunjeong Jeong, Hyesung Kim, Mehdi Bennis, Seong-Lyun Kim .[Mix2FLD: Downlink Federated Learning After Uplink Federated Distillation With Two-Way Mixup](https://arxiv.org/pdf/2006.09801) [J]. arXiv preprint arXiv:2006.09801.
- Tourani R, Srikanteswara S, Misra S, et al. [Democratizing the Edge: A Pervasive Edge Computing Framework](https://arxiv.org/pdf/2007.00641)[J]. arXiv preprint arXiv:2007.00641, 2020.


## System Design
- [Baseline]Brendan McMahan H, Moore E, Ramage D, et al. [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)[J]. arXiv, 2016: arXiv: 1602.05629.
- Ryffel T, Trask A, Dahl M, et al. [A generic framework for privacy preserving deep learning](https://arxiv.org/pdf/1811.04017.pdf!)[J]. arXiv preprint arXiv:1811.04017, 2018.
- [good]Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konecny, Stefano Mazzocchi, H. Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, Jason Roselander .[Towards Federated Learning at Scale: System Design](https://arxiv.org/pdf/1902.01046) [J]. arXiv preprint arXiv:1902.01046.
- Paritosh Ramanan, Kiyoshi Nakayama, Ratnesh Sharma .[BAFFLE : Blockchain based Aggregator Free Federated Learning](https://arxiv.org/pdf/1909.07452) [J]. arXiv preprint arXiv:1909.07452.
- Galtier M N, Marini C. [Substra: a framework for privacy-preserving, traceable and collaborative Machine Learning](https://arxiv.org/pdf/1910.11567)[J]. arXiv preprint arXiv:1910.11567, 2019.
- Anirban Das, Thomas Brunschwiler .[Privacy is What We Care About: Experimental Investigation of Federated Learning on Edge Devices](https://arxiv.org/pdf/1911.04559) [J]. arXiv preprint arXiv:1911.04559.
- Zirui Xu, Zhao Yang, Jinjun Xiong, Jianlei Yang, Xiang Chen .[ELFISH: Resource-Aware Federated Learning on Heterogeneous Edge Devices](https://arxiv.org/pdf/1912.01684) [J]. arXiv preprint arXiv:1912.01684.
- Qinghe Jing, Weiyan Wang, Junxue Zhang, Han Tian, Kai Chen .[Quantifying the Performance of Federated Transfer Learning](https://arxiv.org/pdf/1912.12795) [J]. arXiv preprint arXiv:1912.12795.
- Jiang J, Ji S, Long G. [Decentralized knowledge acquisition for mobile internet applications](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/content/pdf/10.1007/s11280-019-00775-w.pdf&casa_token=n41M2VzZ6UkAAAAA:K-Z0rlst7vi-5s47Hytmmo0N7oRKWglGXJuhjFR200FdUMg_YoeJ8J7e5eT64C2AE4lv61Xu72czRXQ)[J]. World Wide Web, 2020: 1-17.
- Pengchao Han, Shiqiang Wang, Kin K. Leung .[Adaptive Gradient Sparsification for Efficient Federated Learning: An Online Learning Approach](https://arxiv.org/pdf/2001.04756) [J]. arXiv preprint arXiv:2001.04756.
- Zheng Chai, Ahsan Ali, Syed Zawad, Stacey Truex, Ali Anwar, Nathalie Baracaldo, Yi Zhou, Heiko Ludwig, Feng Yan, Yue Cheng .[TiFL: A Tier-based Federated Learning System](https://arxiv.org/pdf/2001.09249) [J]. arXiv preprint arXiv:2001.09249.
- Rongfei Zeng, Shixun Zhang, Jiaqi Wang, Xiaowen Chu .[FMore: An Incentive Scheme of Multi-dimensional Auction for Federated Learning in MEC](https://arxiv.org/pdf/2002.09699) [J]. arXiv preprint arXiv:2002.09699.
- Liu D, Chen X, Zhou Z, et al. [HierTrain: Fast Hierarchical Edge AI Learning with Hybrid Parallelism in Mobile-Edge-Cloud Computing](https://arxiv.org/pdf/2003.09876.pdf)[J]. IEEE Open Journal of the Communications Society, 2020.
- Thomas Hiessl, Daniel Schall, Jana Kemnitz, Stefan Schulte .[Industrial Federated Learning -- Requirements and System Design](https://arxiv.org/pdf/2005.06850) [J]. arXiv preprint arXiv:2005.06850.
- Wu G, Gong S. [Decentralised Learning from Independent Multi-Domain Labels for Person Re-Identification](https://arxiv.org/pdf/2006.04150)[J]. arXiv preprint arXiv:2006.04150, 2020.
- Chengxu Yang, QiPeng Wang, Mengwei Xu, Shangguang Wang, Kaigui Bian, Xuanzhe Liu .[Heterogeneity-Aware Federated Learning](https://arxiv.org/pdf/2006.06983) [J]. arXiv preprint arXiv:2006.06983.
- Georgios Damaskinos, Rachid Guerraoui, Anne-Marie Kermarrec, Vlad Nitu, Rhicheek Patra, Francois Taiani .[FLeet: Online Federated Learning via Staleness Awareness and Performance Prediction](https://arxiv.org/pdf/2006.07273) [J]. arXiv preprint arXiv:2006.07273.
- Nuria Rodríguez-Barroso, Goran Stipcich, Daniel Jiménez-López, José Antonio Ruiz-Millán, Eugenio Martínez-Cámara, Gerardo González-Seco, M. Victoria Luzón, Miguel Ángel Veganzones, Francisco Herrera .[Federated Learning and Differential Privacy: Software tools analysis, the Sherpa.ai FL framework and methodological guidelines for preserving data privacy](https://arxiv.org/pdf/2007.00914) [J]. arXiv preprint arXiv:2007.00914.
- Chaoyang He, Songze Li, Jinhyun So, Mi Zhang, Hongyi Wang, Xiaoyang Wang, Praneeth Vepakomma, Abhishek Singh, Hang Qiu, Li Shen, Peilin Zhao, Yan Kang, Yang Liu, Ramesh Raskar, Qiang Yang, Murali Annavaram, Salman Avestimehr .[FedML: A Research Library and Benchmark for Federated Machine Learning](https://arxiv.org/pdf/2007.13518) [J]. arXiv preprint arXiv:2007.13518.<br>[code:[FedML-AI/FedML](https://github.com/FedML-AI/FedML)]
- Daniel J. Beutel, Taner Topal, Akhil Mathur, Xinchi Qiu, Titouan Parcollet, Nicholas D. Lane .[Flower: A Friendly Federated Learning Research Framework](https://arxiv.org/pdf/2007.14390) [J]. arXiv preprint arXiv:2007.14390.


## Models
- Cho M, Lai L, Xu W. [Distributed Dual Coordinate Ascent in General Tree Networks and Its Application in Federated Learning](https://arxiv.org/pdf/1703.04785)[J]. arXiv preprint arXiv:1703.04785, 2017.
- Hardy C, Le Merrer E, Sericola B. [Md-gan: Multi-discriminator generative adversarial networks for distributed datasets](https://arxiv.org/pdf/1811.03850)[C]//2019 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 2019: 866-877.
- Chen, Qingrong, et al. [Differentially Private Data Generative Models.](https://arxiv.org/pdf/1812.02274) ArXiv Preprint ArXiv:1812.02274, 2018.
- Yang Liu, Yingting Liu, Zhijie Liu, Junbo Zhang, Chuishi Meng, Yu Zheng .[Federated Forest](https://arxiv.org/pdf/1905.10053) [J]. arXiv preprint arXiv:1905.10053.
- Di Chai, Leye Wang, Kai Chen, Qiang Yang .[Secure Federated Matrix Factorization](https://arxiv.org/pdf/1906.05108) [J]. arXiv preprint arXiv:1906.05108.
- Ickin S, Vandikas K, Fiedler M. [Privacy preserving qoe modeling using collaborative learning](https://arxiv.org/pdf/1906.09248)[C]//Proceedings of the 4th Internet-QoE Workshop on QoE-based Analysis and Management of Data Communication Networks. 2019: 13-18.
- Mengwei Yang, Linqi Song, Jie Xu, Congduan Li, Guozhen Tan .[The Tradeoff Between Privacy and Accuracy in Anomaly Detection Using Federated XGBoost](https://arxiv.org/pdf/1907.07157) [J]. arXiv preprint arXiv:1907.07157.
- Seok-Ju Hahn, Junghye Lee .[Privacy-preserving Federated Bayesian Learning of a Generative Model for Imbalanced Classification of Clinical Data](https://arxiv.org/pdf/1910.08489) [J]. arXiv preprint arXiv:1910.08489.
- Qinbin Li, Zeyi Wen, Bingsheng He .[Practical Federated Gradient Boosting Decision Trees](https://arxiv.org/pdf/1911.04206) [J]. arXiv preprint arXiv:1911.04206.
- [ICLR]Augenstein, Sean, et al. [Generative Models for Effective ML on Private, Decentralized Datasets](https://arxiv.org/abs/1911.06679) ArXiv Preprint ArXiv:1911.06679, 2019.<br>[code:[tensorflow/gan](https://github.com/tensorflow/gan)]
- Feng Z, Xiong H, Song C, et al. [SecureGBM: Secure multi-party gradient boosting](https://arxiv.org/pdf/1911.11997)[C]//2019 IEEE International Conference on Big Data (Big Data). IEEE, 2019: 1312-1321.
- Shuai Wang, Tsung-Hui Chang .[Federated Clustering via Matrix Factorization Models: From Model Averaging to Gradient Sharing](https://arxiv.org/pdf/2002.04930) [J]. arXiv preprint arXiv:2002.04930.
- Yang Liu, Mingxin Chen, Wenxi Zhang, Junbo Zhang, Yu Zheng .[Federated Extra-Trees with Privacy Preserving](https://arxiv.org/pdf/2002.07323) [J]. arXiv preprint arXiv:2002.07323.
- Rei Ito, Mineto Tsukada, Hiroki Matsutani .[An On-Device Federated Learning Approach for Cooperative Anomaly Detection](https://arxiv.org/pdf/2002.12301) [J]. arXiv preprint arXiv:2002.12301.
- Chenyou Fan, Ping Liu .[Federated Generative Adversarial Learning](https://arxiv.org/pdf/2005.03793) [J]. arXiv preprint arXiv:2005.03793.
- Mathieu Andreux, Andre Manoel, Romuald Menuet, Charlie Saillard, Chloé Simpson .[Federated Survival Analysis with Discrete-Time Cox Models](https://arxiv.org/pdf/2006.08997) [J]. arXiv preprint arXiv:2006.08997.
- Dashan Gao, Ben Tan, Ce Ju, Vincent W. Zheng, Qiang Yang .[Privacy Threats Against Federated Matrix Factorization](https://arxiv.org/pdf/2007.01587) [J]. arXiv preprint arXiv:2007.01587.



## Natural language Processing
- David Leroy, Alice Coucke, Thibaut Lavril, Thibault Gisselbrecht, Joseph Dureau .[Federated Learning for Keyword Spotting](https://arxiv.org/pdf/1810.05512) [J]. arXiv preprint arXiv:1810.05512.
- [good]Andrew Hard, Kanishka Rao, Rajiv Mathews, Françoise Beaufays, Sean Augenstein, Hubert Eichner, Chloé Kiddon, Daniel Ramage .[Federated Learning for Mobile Keyboard Prediction](https://arxiv.org/pdf/1811.03604) [J]. arXiv preprint arXiv:1811.03604.
- Timothy Yang, Galen Andrew, Hubert Eichner, Haicheng Sun, Wei Li, Nicholas Kong, Daniel Ramage, Françoise Beaufays .[Applied Federated Learning: Improving Google Keyboard Query Suggestions](https://arxiv.org/pdf/1812.02903) [J]. arXiv preprint arXiv:1812.02903.
- Ji S, Pan S, Long G, et al. [Learning private neural language modeling with attentive aggregation](https://arxiv.org/pdf/1812.07108)[C]//2019 International Joint Conference on Neural Networks (IJCNN). IEEE, 2019: 1-8.<br>[code:[shaoxiongji/fed-att](https://github.com/shaoxiongji/fed-att)]
- Jiang, Di, et al. [Federated Topic Modeling](https://dl.acm.org/doi/10.1145/3357384.3357909) Proceedings of the 28th ACM International Conference on Information and Knowledge Management, 2019, pp. 1071–1080.
- Mingqing Chen, Rajiv Mathews, Tom Ouyang, Françoise Beaufays .[Federated Learning Of Out-Of-Vocabulary Words](https://arxiv.org/pdf/1903.10635) [J]. arXiv preprint arXiv:1903.10635.
- Rajagopal. A, Nirmala. V .[Federated AI lets a team imagine together: Federated Learning of GANs](https://arxiv.org/pdf/1906.03595) [J]. arXiv preprint arXiv:1906.03595.
- Swaroop Ramaswamy, Rajiv Mathews, Kanishka Rao, Françoise Beaufays .[Federated Learning for Emoji Prediction in a Mobile Keyboard](https://arxiv.org/pdf/1906.04329) [J]. arXiv preprint arXiv:1906.04329.
- Dianbo Liu, Dmitriy Dligach, Timothy Miller .[Two-stage Federated Phenotyping and Patient Representation Learning](https://arxiv.org/pdf/1908.05596) [J]. arXiv preprint arXiv:1908.05596.
- Duc Bui, Kshitiz Malik, Jack Goetz, Honglei Liu, Seungwhan Moon, Anuj Kumar, Kang G. Shin .[Federated User Representation Learning](https://arxiv.org/pdf/1909.12535) [J]. arXiv preprint arXiv:1909.12535.
- Mingqing Chen, Ananda Theertha Suresh, Rajiv Mathews, Adeline Wong, Cyril Allauzen, Françoise Beaufays, Michael Riley .[Federated Learning of N-gram Language Models](https://arxiv.org/pdf/1910.03432) [J]. arXiv preprint arXiv:1910.03432.
- Florian Hartmann, Sunah Suh, Arkadiusz Komarzewski, Tim D. Smith, Ilana Segall .[Federated Learning for Ranking Browser History Suggestions](https://arxiv.org/pdf/1911.11807) [J]. arXiv preprint arXiv:1911.11807.
- Dianbo Liu, Tim Miller .[Federated pretraining and fine tuning of BERT using clinical notes from multiple silos](https://arxiv.org/pdf/2002.08562) [J]. arXiv preprint arXiv:2002.08562.
- Suyu Ge, Fangzhao Wu, Chuhan Wu, Tao Qi, Yongfeng Huang, Xing Xie .[FedNER: Privacy-preserving Medical Named Entity Recognition with Federated Learning](https://arxiv.org/pdf/2003.09288) [J]. arXiv preprint arXiv:2003.09288.
- Joel Stremmel, Arjun Singh .[Pretraining Federated Text Models for Next Word Prediction](https://arxiv.org/pdf/2005.04828) [J]. arXiv preprint arXiv:2005.04828.
- Om Thakkar, Swaroop Ramaswamy, Rajiv Mathews, Françoise Beaufays .[Understanding Unintended Memorization in Federated Learning](https://arxiv.org/pdf/2006.07490) [J]. arXiv preprint arXiv:2006.07490.


## Computer Vision
- Tzu-Ming Harry Hsu, Hang Qi, Matthew Brown .[Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/pdf/1909.06335) [J]. arXiv preprint arXiv:1909.06335.
- Yang Liu, Anbu Huang, Yun Luo, He Huang, Youzhi Liu, Yuanyuan Chen, Lican Feng, Tianjian Chen, Han Yu, Qiang Yang .[FedVision: An Online Visual Object Detection Platform Powered by Federated Learning](https://arxiv.org/pdf/2001.06202) [J]. arXiv preprint arXiv:2001.06202.
- [CVPR]Tzu-Ming Harry Hsu, Hang Qi, Matthew Brown .[Federated Visual Classification with Real-World Data Distribution](https://arxiv.org/pdf/2003.08082) [J]. arXiv preprint arXiv:2003.08082.<br>[code:[google-research/federated_vision_datasets](https://github.com/google-research/google-research/tree/master/federated_vision_datasets)]
- Rui Shao, Pramuditha Perera, Pong C. Yuen, Vishal M. Patel .[Federated Face Anti-spoofing](https://arxiv.org/pdf/2005.14638) [J]. arXiv preprint arXiv:2005.14638.



## Health Care
- Sheller M J, Reina G A, Edwards B, et al. [Multi-institutional deep learning modeling without sharing patient data: A feasibility study on brain tumor segmentation](https://arxiv.org/abs/1810.04304)[C]//International MICCAI Brainlesion Workshop. Springer, Cham, 2018: 92-104.
- Santiago Silva, Boris Gutman, Eduardo Romero, Paul M Thompson, Andre Altmann, Marco Lorenzi .[Federated Learning in Distributed Medical Databases: Meta-Analysis of Large-Scale Subcortical Brain Data](https://arxiv.org/pdf/1810.08553) [J]. arXiv preprint arXiv:1810.08553.
- Dianbo Liu, Timothy Miller, Raheel Sayeed, Kenneth Mandl .[FADL:Federated-Autonomous Deep Learning for Distributed Electronic Health Record](https://arxiv.org/pdf/1811.11400) [J]. arXiv preprint arXiv:1811.11400.
- Li Huang, Yifeng Yin, Zeng Fu, Shifa Zhang, Hao Deng, Dianbo Liu .[LoAdaBoost:Loss-Based AdaBoost Federated Machine Learning on medical Data](https://arxiv.org/pdf/1811.12629) [J]. arXiv preprint arXiv:1811.12629.
- Li Huang, Dianbo Liu .[Patient Clustering Improves Efficiency of Federated Machine Learning to predict mortality and hospital stay time using distributed Electronic Medical Records](https://arxiv.org/pdf/1903.09296) [J]. arXiv preprint arXiv:1903.09296.
- Yiqiang Chen, Jindong Wang, Chaohui Yu, Wen Gao, Xin Qin .[FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare](https://arxiv.org/pdf/1907.09173) [J]. arXiv preprint arXiv:1907.09173.
- Dashan Gao, Ce Ju, Xiguang Wei, Yang Liu, Tianjian Chen, Qiang Yang .[HHHFL: Hierarchical Heterogeneous Horizontal Federated Learning for Electroencephalography](https://arxiv.org/pdf/1909.05784) [J]. arXiv preprint arXiv:1909.05784.
- Wenqi Li, Fausto Milletarì, Daguang Xu, Nicola Rieke, Jonny Hancox, Wentao Zhu, Maximilian Baust, Yan Cheng, Sébastien Ourselin, M. Jorge Cardoso, Andrew Feng .[Privacy-preserving Federated Brain Tumour Segmentation](https://arxiv.org/pdf/1910.00962) [J]. arXiv preprint arXiv:1910.00962.
- [good]Dianbo Liu, Timothy A Miller, Kenneth D. Mandl .[Confederated Machine Learning on Horizontally and Vertically Separated Medical Data for Large-Scale Health System Intelligence](https://arxiv.org/pdf/1910.02109) [J]. arXiv preprint arXiv:1910.02109.
- Rulin Shao, Hui Liu, Dianbo Liu .[Privacy Preserving Stochastic Channel-Based Federated Learning with Neural Network Pruning](https://arxiv.org/pdf/1910.02115) [J]. arXiv preprint arXiv:1910.02115.
- Olivia Choudhury, Aris Gkoulalas-Divanis, Theodoros Salonidis, Issa Sylla, Yoonyoung Park, Grace Hsu, Amar Das .[Differential Privacy-enabled Federated Learning for Sensitive Health Data](https://arxiv.org/pdf/1910.02578) [J]. arXiv preprint arXiv:1910.02578.
- Sabri Boughorbel, Fethi Jarray, Neethu Venugopal, Shabir Moosa, Haithum Elhadi, Michel Makhlouf .[Federated Uncertainty-Aware Learning for Distributed Hospital EHR Data](https://arxiv.org/pdf/1910.12191) [J]. arXiv preprint arXiv:1910.12191.
- Jonathan Passerat-Palmbach, Tyler Farnan, Robert Miller, Marielle S. Gross, Heather Leigh Flannery, Bill Gleim .[A blockchain-orchestrated Federated Learning architecture for healthcare consortia](https://arxiv.org/pdf/1910.12603) [J]. arXiv preprint arXiv:1910.12603.
- Stephen R. Pfohl, Andrew M. Dai, Katherine Heller .[Federated and Differentially Private Learning for Electronic Health Records](https://arxiv.org/pdf/1911.05861) [J]. arXiv preprint arXiv:1911.05861.
- Jie Xu, Fei Wang .[Federated Learning for Healthcare Informatics](https://arxiv.org/pdf/1911.06270) [J]. arXiv preprint arXiv:1911.06270.
- Sharma P, Shamout F E, Clifton D A. [Preserving patient privacy while training a predictive model of in-hospital mortality](https://arxiv.org/pdf/1912.00354)[J]. arXiv preprint arXiv:1912.00354, 2019.
- Songtao Lu, Yawen Zhang, Yunlong Wang, Christina Mack .[Learn Electronic Health Records by Fully Decentralized Federated Learning](https://arxiv.org/pdf/1912.01792) [J]. arXiv preprint arXiv:1912.01792.
- Xiaoxiao Li, Yufeng Gu, Nicha Dvornek, Lawrence Staib, Pamela Ventola, James S. Duncan .[Multi-site fMRI Analysis Using Privacy-preserving Federated Learning and Domain Adaptation: ABIDE Results](https://arxiv.org/pdf/2001.05647) [J]. arXiv preprint arXiv:2001.05647.
- R. Bey, R. Goussault, M. Benchoufi, R. Porcher .[Stratified cross-validation for unbiased and privacy-preserving federated learning](https://arxiv.org/pdf/2001.08090) [J]. arXiv preprint arXiv:2001.08090.
- Jianfei Cui, Dianbo Liu .[Federated machine learning with Anonymous Random Hybridization (FeARH) on medical records](https://arxiv.org/pdf/2001.09751) [J]. arXiv preprint arXiv:2001.09751.
- Olivia Choudhury, Aris Gkoulalas-Divanis, Theodoros Salonidis, Issa Sylla, Yoonyoung Park, Grace Hsu, Amar Das .[Anonymizing Data for Privacy-Preserving Federated Learning](https://arxiv.org/pdf/2002.09096) [J]. arXiv preprint arXiv:2002.09096.
- Nicola Rieke, Jonny Hancox, Wenqi Li, Fausto Milletari, Holger Roth, Shadi Albarqouni, Spyridon Bakas, Mathieu N. Galtier, Bennett Landman, Klaus Maier-Hein, Sebastien Ourselin, Micah Sheller, Ronald M. Summers, Andrew Trask, Daguang Xu, Maximilian Baust, M. Jorge Cardoso .[The Future of Digital Health with Federated Learning](https://arxiv.org/pdf/2003.08119) [J]. arXiv preprint arXiv:2003.08119.
- Ce Ju, Dashan Gao, Ravikiran Mane, Ben Tan, Yang Liu, Cuntai Guan .[Federated Transfer Learning for EEG Signal Classification](https://arxiv.org/pdf/2004.12321) [J]. arXiv preprint arXiv:2004.12321.
- Binhang Yuan, Song Ge, Wenhui Xing .[A Federated Learning Framework for Healthcare IoT devices](https://arxiv.org/pdf/2005.05083) [J]. arXiv preprint arXiv:2005.05083.
- Ce Ju, Ruihui Zhao, Jichao Sun, Xiguang Wei, Bo Zhao, Yang Liu, Hongshan Li, Tianjian Chen, Xinwei Zhang, Dashan Gao, Ben Tan, Han Yu, Yuan Jin .[Privacy-Preserving Technology to Help Millions of People: Federated Prediction Model for Stroke Prevention](https://arxiv.org/pdf/2006.10517) [J]. arXiv preprint arXiv:2006.10517.


## Transportation
- Sumudu Samarakoon, Mehdi Bennis, Walid Saad, Merouane Debbah .[Federated Learning for Ultra-Reliable Low-Latency V2V Communications](https://arxiv.org/pdf/1805.09253) [J]. arXiv preprint arXiv:1805.09253.
- Sumudu Samarakoon, Mehdi Bennis, Walid Saad, Merouane Debbah .[Distributed Federated Learning for Ultra-Reliable Low-Latency Vehicular  Communications](https://arxiv.org/pdf/1807.08127) [J]. arXiv preprint arXiv:1807.08127.
- Yuris Mulya Saputra, Dinh Thai Hoang, Diep N. Nguyen, Eryk Dutkiewicz, Markus Dominik Mueck, Srikathyayani Srikanteswara .[Energy Demand Prediction with Federated Learning for Electric Vehicle Networks](https://arxiv.org/pdf/1909.00907) [J]. arXiv preprint arXiv:1909.00907.
- Xinle Liang, Yang Liu, Tianjian Chen, Ming Liu, Qiang Yang .[Federated Transfer Reinforcement Learning for Autonomous Driving](https://arxiv.org/pdf/1910.06001) [J]. arXiv preprint arXiv:1910.06001.
- Ye D, Yu R, Pan M, et al. [Federated learning in vehicular edge computing: A selective model aggregation approach](https://ieeexplore.ieee.org/iel7/6287639/8948470/08964354.pdf)[J]. IEEE Access, 2020, 8: 23920-23935.
- Bekir Sait Ciftler, Abdullatif Albaseer, Noureddine Lasla, Mohamed Abdallah .[Federated Learning for Localization: A Privacy-Preserving Crowdsourcing Method](https://arxiv.org/pdf/2001.01911) [J]. arXiv preprint arXiv:2001.01911.
- Chaochao Chen, Jun Zhou, Bingzhe Wu, Wenjin Fang, Li Wang, Yuan Qi, Xiaolin Zheng. [Practical Privacy Preserving POI Recommendation](https://arxiv.org/pdf/2003.02834.pdf) [J]. arXiv preprint arXiv:2003.02834.
- Feng Yin, Zhidi Lin, Yue Xu, Qinglei Kong, Deshi Li, Sergios Theodoridis, Shuguang (Robert)Cui .[FedLoc: Federated Learning Framework for Cooperative Localization and Location Data Processing](https://arxiv.org/pdf/2003.03697) [J]. arXiv preprint arXiv:2003.03697.
- Hamid Shiri, Jihong Park, Mehdi Bennis .[Communication-Efficient Massive UAV Online Path Control: Federated Learning Meets Mean-Field Game Theory](https://arxiv.org/pdf/2003.04451) [J]. arXiv preprint arXiv:2003.04451.
- Yi Liu, James J.Q. Yu, Jiawen Kang, Dusit Niyato, Shuyu Zhang .[Privacy-preserving Traffic Flow Prediction: A Federated Learning Approach](https://arxiv.org/pdf/2003.08725) [J]. arXiv preprint arXiv:2003.08725.
- van Hulst J M, Zeni M, Kröller A, et al. [Beyond privacy regulations: an ethical approach to data usage in transportation](https://arxiv.org/pdf/2004.00491)[J]. arXiv preprint arXiv:2004.00491, 2020.
- Yuris Mulya Saputra, Diep N. Nguyen, Dinh Thai Hoang, Thang Xuan Vu, Eryk Dutkiewicz, Symeon Chatzinotas .[Federated Learning Meets Contract Theory: Energy-Efficient Framework for Electric Vehicle Networks](https://arxiv.org/pdf/2004.01828) [J]. arXiv preprint arXiv:2004.01828.
- Wei Yang Bryan Lim, Jianqiang Huang, Zehui Xiong, Jiawen Kang, Dusit Niyato, Xian-Sheng Hua, Cyril Leung, Chunyan Miao .[Towards Federated Learning in UAV-Enabled Internet of Vehicles: A Multi-Dimensional Contract-Matching Approach](https://arxiv.org/pdf/2004.03877) [J]. arXiv preprint arXiv:2004.03877.
- Ahmet M. Elbir, S. Coleri .[Federated Learning for Vehicular Networks](https://arxiv.org/pdf/2006.01412) [J]. arXiv preprint arXiv:2006.01412.


## Recommendation System
- Fei Chen, Zhenhua Dong, Zhenguo Li, Xiuqiang He .[Federated Meta-Learning for Recommendation](https://arxiv.org/pdf/1802.07876) [J]. arXiv preprint arXiv:1802.07876.
- Muhammad Ammad-ud-din, Elena Ivannikova, Suleiman A. Khan, Were Oyomno, Qiang Fu, Kuan Eeik Tan, Adrian Flanagan .[Federated Collaborative Filtering for Privacy-Preserving Personalized Recommendation System](https://arxiv.org/pdf/1901.09888) [J]. arXiv preprint arXiv:1901.09888.
- Di Chai, Leye Wang, Kai Chen, Qiang Yang .[Secure Federated Matrix Factorization](https://arxiv.org/pdf/1906.05108) [J]. arXiv preprint arXiv:1906.05108.
- Feng Liao, Hankz Hankui Zhuo, Xiaoling Huang, Yu Zhang .[Federated Hierarchical Hybrid Networks for Clickbait Detection](https://arxiv.org/pdf/1906.00638) [J]. arXiv preprint arXiv:1906.00638.
- Lin Y, Ren P, Chen Z, et al. [Meta Matrix Factorization for Federated Rating Predictions](https://arxiv.org/pdf/1910.10086.pdf)[C]//Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2020: 981-990.
- Ribero M, Henderson J, Williamson S, et al. [Federating Recommendations Using Differentially Private Prototypes](https://arxiv.org/pdf/2003.00602)[J]. arXiv preprint arXiv:2003.00602, 2020.
- Tao Qi, Fangzhao Wu, Chuhan Wu, Yongfeng Huang, Xing Xie .[FedRec: Privacy-Preserving News Recommendation with Federated Learning](https://arxiv.org/pdf/2003.09592) [J]. arXiv preprint arXiv:2003.09592.
- Adrian Flanagan, Were Oyomno, Alexander Grigorievskiy, Kuan Eeik Tan, Suleiman A. Khan, Muhammad Ammad-Ud-Din .[Federated Multi-view Matrix Factorization for Personalized Recommendations](https://arxiv.org/pdf/2004.04256) [J]. arXiv preprint arXiv:2004.04256.
- Tan Li, Linqi Song, Christina Fragouli .[Federated Recommendation System via Differential Privacy](https://arxiv.org/pdf/2005.06670) [J]. arXiv preprint arXiv:2005.06670.
- Chen Chen, Jingfeng Zhang, Anthony K. H. Tung, Mohan Kankanhalli, Gang Chen .[Robust Federated Recommendation System](https://arxiv.org/pdf/2006.08259) [J]. arXiv preprint arXiv:2006.08259.


## Speech Recognition
- Andrew Hard, Kurt Partridge, Cameron Nguyen, Niranjan Subrahmanya, Aishanee Shah, Pai Zhu, Ignacio Lopez Moreno, Rajiv Mathews .[Training Keyword Spotting Models on Non-IID Data with Federated Learning](https://arxiv.org/pdf/2005.10406) [J]. arXiv preprint arXiv:2005.10406.


## Finance && Blockchain 
- Hyesung Kim, Jihong Park, Mehdi Bennis, Seong-Lyun Kim .[On-Device Federated Learning via Blockchain and its Latency Analysis](https://arxiv.org/pdf/1808.03949) [J]. arXiv preprint arXiv:1808.03949.
- Toyotaro Suzumura, Yi Zhou, Natahalie Baracaldo, Guangnan Ye, Keith Houck, Ryo Kawahara, Ali Anwar, Lucia Larise Stavarache, Yuji Watanabe, Pablo Loyola, Daniel Klyashtorny, Heiko Ludwig, Kumar Bhaskaran .[Towards Federated Graph Learning for Collaborative Financial Crimes Detection](https://arxiv.org/pdf/1909.12946) [J]. arXiv preprint arXiv:1909.12946.
- Yuan Liu, Shuai Sun, Zhengpeng Ai, Shuangfeng Zhang, Zelei Liu, Han Yu .[FedCoin: A Peer-to-Peer Payment System for Federated Learning](https://arxiv.org/pdf/2002.11711) [J]. arXiv preprint arXiv:2002.11711.


## Smart City &&  Other Applications
- Nguyen T D, Marchal S, Miettinen M, et al. [DIoT: A federated self-learning anomaly detection system for IoT](https://arxiv.org/pdf/1804.07474.pdf)[C]//2019 IEEE 39th International Conference on Distributed Computing Systems (ICDCS). IEEE, 2019: 756-767.
- Yujing Chen, Yue Ning, Zheng Chai, Huzefa Rangwala .[Federated Multi-task Hierarchical Attention Model for Sensor Analytics](https://arxiv.org/pdf/1905.05142) [J]. arXiv preprint arXiv:1905.05142.
- Tagliasacchi M, Gfeller B, Quitry F C, et al. [Self-supervised audio representation learning for mobile devices](https://arxiv.org/pdf/1907.10218.pdf)[J]. arXiv preprint arXiv:1905.11796, 2019.
- Feng J, Rong C, Sun F, et al. [Pmf: A privacy-preserving human mobility prediction framework via federated learning](https://vonfeng.github.io/files/UbiComp2020_PMF_Final.pdf)[J]. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 2020, 4(1): 1-21.
- Abdullatif Albaseer, Bekir Sait Ciftler, Mohamed Abdallah, Ala Al-Fuqaha .[Exploiting Unlabeled Data in Smart Cities using Federated Learning](https://arxiv.org/pdf/2001.04030) [J]. arXiv preprint arXiv:2001.04030.
- Nicolas Aussel (INF, ACMES-SAMOVAR, IP Paris), Sophie Chabridon (IP Paris, INF, ACMES-SAMOVAR), Yohan Petetin (TIPIC-SAMOVAR, CITI, IP Paris) .[Combining Federated and Active Learning for Communication-efficient Distributed Failure Prediction in Aeronautics](https://arxiv.org/pdf/2001.07504) [J]. arXiv preprint arXiv:2001.07504.
- Zhuzhu Wang, Yilong Yang, Yang Liu, Ximeng Liu, Brij B. Gupta, Jianfeng Ma .[Cloud-based Federated Boosting for Mobile Crowdsensing](https://arxiv.org/pdf/2005.05304) [J]. arXiv preprint arXiv:2005.05304.


## Uncategorized
### 2015
- Shokri R, Shmatikov V. [Privacy-preserving deep learning](http://www.cs.cornell.edu/~shmat/shmat_ccs15.pdf)[C]//Proceedings of the 22nd ACM SIGSAC conference on computer and communications security. 2015: 1310-1321.

### 2016
- Abadi M, Chu A, Goodfellow I, et al. [Deep Learning with Differential Privacy](https://arxiv.org/pdf/1607.00133.pdf)[J]. arXiv preprint arXiv:1607.00133, 2016.
- Shokri R, Stronati M, Song C, et al. [Membership inference attacks against machine learning models](https://arxiv.org/pdf/1610.05820)[C]//2017 IEEE Symposium on Security and Privacy (SP). IEEE, 2017: 3-18.<br>[code:[csong27/membership-inference](https://github.com/csong27/membership-inference)]


### 2017
- Mohassel P, Zhang Y. Secureml: [A system for scalable privacy-preserving machine learning](http://web.eecs.umich.edu/~mosharaf/Readings/SecureML.pdf)[C]//2017 IEEE Symposium on Security and Privacy (SP). IEEE, 2017: 19-38.
- Md Nazmus Sadat, Md Momin Al Aziz, Noman Mohammed, Feng Chen, Shuang Wang, Xiaoqian Jiang .[SAFETY: Secure gwAs in Federated Environment Through a hYbrid solution  with Intel SGX and Homomorphic Encryption](https://arxiv.org/pdf/1703.02577) [J]. arXiv preprint arXiv:1703.02577.
- [KDD]Yejin Kim, Jimeng Sun, Hwanjo Yu, Xiaoqian Jiang .[Federated Tensor Factorization for Computational Phenotyping](https://arxiv.org/pdf/1704.03141) [J]. arXiv preprint arXiv:1704.03141.
- Xu Jiang, Nan Guan, Xiang Long, Wang Yi .[Semi-Federated Scheduling of Parallel Real-Time Tasks on Multiprocessors](https://arxiv.org/pdf/1705.03245) [J]. arXiv preprint arXiv:1705.03245.
- Gabriela Montoya, Hala Skaf-Molli, Katja Hose .[The Odyssey Approach for Optimizing Federated SPARQL Queries](https://arxiv.org/pdf/1705.06135) [J]. arXiv preprint arXiv:1705.06135.
- Benedicto B. Balilo Jr., Bobby D. Gerardo, Ruji P. Medina, Yungcheol Byun .[A Unique One-Time Password Table Sequence Pattern Authentication:  Application to Bicol University Union of Federated Faculty Association, Inc.  (BUUFFAI) eVoting System](https://arxiv.org/pdf/1708.00562) [J]. arXiv preprint arXiv:1708.00562.
- Chang K, Balachandar N, Lam C K, et al. [Institutionally Distributed Deep Learning Networks](https://arxiv.org/pdf/1709.05929)[J]. arXiv preprint arXiv:1709.05929, 2017.
- Niklas Ueter, Georg von der Brüggen, Jian-Jia Chen, Jing Li, Kunal Agrawal .[Reservation-Based Federated Scheduling for Parallel Real-Time Tasks](https://arxiv.org/pdf/1712.05040) [J]. arXiv preprint arXiv:1712.05040.
- Saurabh Kumar, Pararth Shah, Dilek Hakkani-Tur, Larry Heck .[Federated Control with Hierarchical Multi-Agent Deep Reinforcement  Learning](https://arxiv.org/pdf/1712.08266) [J]. arXiv preprint arXiv:1712.08266.


### 2018
- Yu Z, Hu J, Min G, et al. [Federated learning based proactive content caching in edge computing](https://ore.exeter.ac.uk/repository/bitstream/handle/10871/36227/Globecom_2018.pdf?sequence=1)[C]//2018 IEEE Global Communications Conference (GLOBECOM). IEEE, 2018: 1-6.
- Wang S, Tuor T, Salonidis T, et al. [When edge meets learning: Adaptive control for resource-constrained distributed machine learning](https://dsprdpub.cc.ic.ac.uk:8443/bitstream/10044/1/58765/2/Infocom_2018_distributed_ML.pdf)[C]//IEEE INFOCOM 2018-IEEE Conference on Computer Communications. IEEE, 2018: 63-71.
- Caldas S, Smith V, Talwalkar A. [Federated Kernelized Multi-Task Learning](https://systemsandml.org/Conferences/2019/doc/2018/30.pdf)[C]//SysML Conference 2018. 2018.
- Juvekar C, Vaikuntanathan V, Chandrakasan A. [{GAZELLE}: A low latency framework for secure neural network inference](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-juvekar.pdf)[C]//27th {USENIX} Security Symposium ({USENIX} Security 18). 2018: 1651-1669.
- [ICLR]Popov V, Kudinov M, Piontkovskaya I, et al. [Distributed fine-tuning of language models on private data](https://openreview.net/pdf?id=HkgNdt26Z)[C]//International Conference on Learning Representations. 2018.
-  .[Improving Privacy and Trust in Federated Identity Using SAML with Hash  Based Encryption Algorithm](https://arxiv.org/pdf/1803.02891) [J]. arXiv preprint arXiv:1803.02891.
- Milosz Pacholczyk, Krzysztof Rzadca
 .[Fair non-monetary scheduling in federated clouds](https://arxiv.org/pdf/1803.06178) [J]. arXiv preprint arXiv:1803.06178.
- Ronghua Xu, Yu Chen, Erik Blasch, Genshe Chen .[A Federated Capability-based Access Control Mechanism for Internet of  Things (IoTs)](https://arxiv.org/pdf/1805.00825) [J]. arXiv preprint arXiv:1805.00825.
- Orekondy T, Oh S J, Zhang Y, et al. [Gradient-Leaks: Understanding and Controlling Deanonymization in Federated Learning](https://arxiv.org/pdf/1805.05838)[J]. arXiv preprint arXiv:1805.05838, 2018.
- Loïc Baron (NPA, CNRS), Radomir Klacza (UPMC, NPA), Pauline Gaudet-Chardonnet (NPA, UPMC), Amira Bradai (UPMC, NPA), Ciro Scognamiglio (UPMC, NPA), Serge Fdida (NPA, LINCS) .[Next generation portal for federated testbeds MySlice v2: from prototype  to production](https://arxiv.org/pdf/1806.04467) [J]. arXiv preprint arXiv:1806.04467.
- Christopher Rackauckas, Qing Nie .[Confederated Modular Differential Equation APIs for Accelerated  Algorithm Development and Benchmarking](https://arxiv.org/pdf/1807.06430) [J]. arXiv preprint arXiv:1807.06430.
- Pablo Orviz Fernandez, Joao Pina, Alvaro Lopez Garcia, Isabel Campos Plasencia, Mario David, Jorge Gomes .[umd-verification: Automation of Software Validation for the EGI  federated e-Infrastructure](https://arxiv.org/pdf/1807.11318) [J]. arXiv preprint arXiv:1807.11318.
- Tobias Grubenmann, Abraham Bernstein, Dmitry Moor, Sven Seuken .[FedMark: A Marketplace for Federated Data on the Web](https://arxiv.org/pdf/1808.06298) [J]. arXiv preprint arXiv:1808.06298.
- Phong, Le Trieu, and Tran Thi Phuong. [Privacy-Preserving Deep Learning via Weight Transmission.](https://arxiv.org/abs/1809.03272) IEEE Transactions on Information Forensics and Security, vol. 14, no. 11, 2019, pp. 3003–3015.
- Dinesh Verma, Simon Julier, Greg Cirincione .[Federated AI for building AI Solutions across Multiple Agencies](https://arxiv.org/pdf/1809.10036) [J]. arXiv preprint arXiv:1809.10036.
- Qijun Zhu, Dandan Li, Dik Lun Lee .[C-DLSI: An Extended LSI Tailored for Federated Text Retrieval](https://arxiv.org/pdf/1810.02579) [J]. arXiv preprint arXiv:1810.02579.
- John Sherlock, Manoj Muniswamaiah, Lauren Clarke, Shawn Cicoria .[Review of Barriers for Federated Identity Adoption for Users and Organizations](https://arxiv.org/pdf/1810.06152) [J]. arXiv preprint arXiv:1810.06152.
- Thanos Yannakis, Pavlos Fafalios, Yannis Tzitzikas .[Heuristics-based Query Reordering for Federated Queries in SPARQL 1.1 and SPARQL-LD](https://arxiv.org/pdf/1810.09780) [J]. arXiv preprint arXiv:1810.09780.
- Álvaro García-Pérez, Alexey Gotsman .[Federated Byzantine Quorum Systems (Extended Version)](https://arxiv.org/pdf/1811.03642) [J]. arXiv preprint arXiv:1811.03642.
- Jan Trienes, Andrés Torres Cano, Djoerd Hiemstra .[Recommending Users: Whom to Follow on Federated Social Networks](https://arxiv.org/pdf/1811.09292) [J]. arXiv preprint arXiv:1811.09292.


### 2019
- Ren J, Wang H, Hou T, et al. [Federated learning-based computation offloading optimization in edge computing-supported internet of things](https://ieeexplore.ieee.org/iel7/6287639/8600701/08728285.pdf)[J]. IEEE Access, 2019, 7: 69194-69201.
- Díaz González F. [Federated Learning for Time Series Forecasting Using LSTM Networks: Exploiting Similarities Through Clustering](https://www.diva-portal.org/smash/get/diva2:1334598/FULLTEXT01.pdf)[J]. 2019.
- Yang W, Zhang Y, Ye K, et al. [FFD: A Federated Learning Based Method for Credit Card Fraud Detection](https://link.springer.com/chapter/10.1007/978-3-030-23551-2_2)[C]//International Conference on Big Data. Springer, Cham, 2019: 18-32.
- Yao X, Huang T, Wu C, et al. [Towards faster and better federated learning: A feature fusion approach](https://ieeexplore.ieee.org/abstract/document/8803001/)[C]//2019 IEEE International Conference on Image Processing (ICIP). IEEE, 2019: 175-179.
- Lu S, Yao Y, Shi W. [Collaborative learning on the edges: A case study on connected vehicles](https://www.usenix.org/system/files/hotedge19-paper-lu.pdf)[C]//2nd {USENIX} Workshop on Hot Topics in Edge Computing (HotEdge 19). 2019.
- Li Y. [Federated Learning for Time Series Forecasting Using Hybrid Model](https://www.diva-portal.org/smash/get/diva2:1334629/FULLTEXT01.pdf)[J]. 2019.
- Anusha Lalitha, Osman Cihan Kilinc, Tara Javidi, Farinaz Koushanfar .[Peer-to-peer Federated Learning on Graphs](https://arxiv.org/pdf/1901.11173) [J]. arXiv preprint arXiv:1901.11173.
- Łukasz Lachowski .[Complexity of the quorum intersection property of the Federated Byzantine Agreement System](https://arxiv.org/pdf/1902.06493) [J]. arXiv preprint arXiv:1902.06493.
- Wennan Zhu, Peter Kairouz, Haicheng Sun, Brendan McMahan, Wei Li .[Federated Heavy Hitters Discovery with Differential Privacy](https://arxiv.org/pdf/1902.08534) [J]. arXiv preprint arXiv:1902.08534.
- Oussama Habachi, Mohamed-Ali Adjif, Jean-Pierre Cances .[Fast Uplink Grant for NOMA: a Federated Learning based Approach](https://arxiv.org/pdf/1904.07975) [J]. arXiv preprint arXiv:1904.07975.
- Sunny Sanyal, Dapeng Wu, Boubakr Nour .[A Federated Filtering Framework for Internet of Medical Things](https://arxiv.org/pdf/1905.01138) [J]. arXiv preprint arXiv:1905.01138.
- Sumit Kumar Monga, Sheshadri K R, Yogesh Simmhan .[ElfStore: A Resilient Data Storage Service for Federated Edge and Fog Resources](https://arxiv.org/pdf/1905.08932) [J]. arXiv preprint arXiv:1905.08932.
- Thomas Hardjono .[A Federated Authorization Framework for Distributed Personal Data and Digital Identity](https://arxiv.org/pdf/1906.03552) [J]. arXiv preprint arXiv:1906.03552.
- Bob Iannucci, Aviral Shrivastava, Mohammad Khayatian .[TickTalk -- Timing API for Dynamically Federated Cyber-Physical Systems](https://arxiv.org/pdf/1906.03982) [J]. arXiv preprint arXiv:1906.03982.
- Nishant Saurabh, Dragi Kimovski, Simon Ostermann, Radu Prodan .[VM Image Repository and Distribution Models for Federated Clouds: State of the Art, Possible Directions and Open Issues](https://arxiv.org/pdf/1906.09182) [J]. arXiv preprint arXiv:1906.09182.
- Peter Mell, Jim Dray, James Shook .[Smart Contract Federated Identity Management without Third Party Authentication Services](https://arxiv.org/pdf/1906.11057) [J]. arXiv preprint arXiv:1906.11057.
- Maria L. B. A. Santos, Jessica C. Carneiro, Antonio M. R. Franco, Fernando A. Teixeira, Marco A. Henriques, Leonardo B. Oliveira .[A Federated Lightweight Authentication Protocol for the Internet of Things](https://arxiv.org/pdf/1907.05527) [J]. arXiv preprint arXiv:1907.05527.
- Andreas Grammenos, Rodrigo Mendoza-Smith, Cecilia Mascolo, Jon Crowcroft .[Federated PCA with Adaptive Rank Estimation](https://arxiv.org/pdf/1907.08059) [J]. arXiv preprint arXiv:1907.08059.
- Andrew Prout, William Arcand, David Bestor, Bill Bergeron, Chansup Byun, Vijay Gadepally, Michael Houle, Matthew Hubbell, Michael Jones, Anna Klein, Peter Michaleas, Lauren Milechin, Julie Mullen, Antonio Rosa, Siddharth Samsi, Charles Yee, Albert Reuther, Jeremy Kepner .[Securing HPC using Federated Authentication](https://arxiv.org/pdf/1908.07573) [J]. arXiv preprint arXiv:1908.07573.
- [ICLR]Li J, Khodak M, Caldas S, et al. [Differentially private meta-learning](https://arxiv.org/pdf/1909.05830)[J]. arXiv preprint arXiv:1909.05830, 2019.
- Sharma V, Vepakomma P, Swedish T, et al. [ExpertMatcher: Automating ML Model Selection for Users in Resource Constrained Countries](https://arxiv.org/pdf/1910.02312)[J]. arXiv preprint arXiv:1910.02312, 2019.
- Rulin Shao, Hongyu He, Hui Liu, Dianbo Liu .[Stochastic Channel-Based Federated Learning for Medical Data Privacy Preserving](https://arxiv.org/pdf/1910.11160) [J]. arXiv preprint arXiv:1910.11160.
- Shashi Raj Pandey, Nguyen H. Tran, Mehdi Bennis, Yan Kyaw Tun, Aunas Manzoor, Choong Seon Hong .[A Crowdsourcing Framework for On-Device Federated Learning](https://arxiv.org/pdf/1911.01046) [J]. arXiv preprint arXiv:1911.01046.
- Kun Ma, Antoine Bagula, Olasupo Ajayi .[Quality of Service (QoS) Modelling in Federated Cloud Computing](https://arxiv.org/pdf/1911.03051) [J]. arXiv preprint arXiv:1911.03051.
- Daniele D'Agostino, Luca Roverelli, Gabriele Zereik, Giuseppe La Rocca, Andrea De Luca, Ruben Salvaterra, Andrea Belfiore, Gianni Lisini, Giovanni Novara, Andrea Tiengo .[A science gateway for Exploring the X-ray Transient and variable sky using EGI Federated Cloud](https://arxiv.org/pdf/1911.06560) [J]. arXiv preprint arXiv:1911.06560.
- André Gaul, Ismail Khoffi, Jörg Liesen, Torsten Stüber .[Mathematical Analysis and Algorithms for Federated Byzantine Agreement Systems](https://arxiv.org/pdf/1912.01365) [J]. arXiv preprint arXiv:1912.01365.
- Xidi Qu, Shengling Wang, Qin Hu, Xiuzhen Cheng .[Proof of Federated Learning: A Novel Energy-recycling Consensus Algorithm](https://arxiv.org/pdf/1912.11745) [J]. arXiv preprint arXiv:1912.11745.
- Boyi Liu, Lujia Wang, Ming Liu, Cheng-Zhong Xu .[Federated Imitation Learning: A Novel Framework for Cloud Robotic Systems with Heterogeneous Sensor Data](https://arxiv.org/pdf/1912.12204) [J]. arXiv preprint arXiv:1912.12204.
- Zhaoxian Wu, Qing Ling, Tianyi Chen, Georgios B. Giannakis .[Federated Variance-Reduced Stochastic Gradient Descent with Robustness to Byzantine Attacks](https://arxiv.org/pdf/1912.12716) [J]. arXiv preprint arXiv:1912.12716.


### 2020
- Ye D, Yu R, Pan M, et al. [Federated learning in vehicular edge computing: A selective model aggregation approach](https://ieeexplore.ieee.org/iel7/6287639/8948470/08964354.pdf)[J]. IEEE Access, 2020, 8: 23920-23935.
- Wang X, Wang C, Li X, et al. [Federated deep reinforcement learning for internet of things with decentralized cooperative edge caching](http://www.mosaic-lab.org/uploads/papers/169d0b9c-0c8f-441f-9600-0c04c0afc8e1.pdf)[J]. IEEE Internet of Things Journal, 2020.
- Semwal T, Mulay A, Agrawal A M. [FedPerf: A Practitioners’ Guide to Performance of Federated Learning Algorithms](https://osf.io/q3vkt/download?format=pdf)[J]. 2020.
- [ICML][communication]Hamer, Jenny, et al. [FedBoost: A Communication-Efficient Algorithm for Federated Learning.](https://proceedings.icml.cc/static/paper_files/icml/2020/5967-Paper.pdf) ICML 2020: 37th International Conference on Machine Learning, vol. 1, 2020.<br>[video:[fedboost-a-communicationefficient-algorithm-for-federated-learning](https://slideslive.com/38928463/fedboost-a-communicationefficient-algorithm-for-federated-learning?ref=speaker-16993-latest)]
- [NIPS][non-I.I.D, personalization]Fallah A, Mokhtari A, Ozdaglar A. [Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](http://proceedings.neurips.cc/paper/2020/file/24389bfe4fe2eba8bf9aa9203a44cdad-Paper.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.
- [NIPS]Dubey A, Pentland A S. [Differentially-Private Federated Linear Bandits](http://proceedings.neurips.cc/paper/2020/file/4311359ed4969e8401880e3c1836fbe1-Paper.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.<br> [code:[abhimanyudubey/private_federated_linear_bandits](https://github.com/abhimanyudubey/private_federated_linear_bandits)]
- [NIPS]Grammenos A, Mendoza Smith R, Crowcroft J, et al. [Federated Principal Component Analysis](https://papers.nips.cc/paper/2020/file/47a658229eb2368a99f1d032c8848542-Paper.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.<br>[code:[andylamp/federated_pca](https://github.com/andylamp/federated_pca)]
- [NIPS][Privacy]Deng Y, Kamani M M, Mahdavi M. [Distributionally Robust Federated Averaging](https://proceedings.neurips.cc/paper/2020/file/ac450d10e166657ec8f93a1b65ca1b14-Paper.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.<br>[code:[MLOPTPSU/FedTorch](https://github.com/MLOPTPSU/FedTorch)]
- [NIPS]He C, Annavaram M, Avestimehr S. [Group Knowledge Transfer: Federated Learning of Large CNNs at the Edge](http://proceedings.neurips.cc/paper/2020/file/a1d4c20b182ad7137ab3606f0e3fc8a4-Paper.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.<br>[code:[FedML-AI/FedML/tree/master/fedml_experiments/distributed/fedgkt](https://github.com/FedML-AI/FedML/tree/master/fedml_experiments/distributed/fedgkt)]
- [NIPS]So J, Guler B, Avestimehr S. [A Scalable Approach for Privacy-Preserving Collaborative Machine Learning](http://proceedings.neurips.cc/paper/2020/file/5bf8aaef51c6e0d363cbe554acaf3f20-Paper.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.
- [NIPS]Chen X, Chen T, Sun H, et al. [Distributed training with heterogeneous data: Bridging median-and mean-based algorithms](https://proceedings.neurips.cc/paper/2020/file/f629ed9325990b10543ab5946c1362fb-Paper.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.
- [NIPS]Bistritz I, Mann A, Bambos N. [Distributed Distillation for On-Device Learning](https://proceedings.neurips.cc/paper/2020/file/fef6f971605336724b5e6c0c12dc2534-Paper.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.
- [NIPS]Li J, Abbas W, Koutsoukos X. [Byzantine Resilient Distributed Multi-Task Learning](http://proceedings.neurips.cc/paper/2020/file/d37eb50d868361ea729bb4147eb3c1d8-Paper.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.
- [NIPS]Ghosh A, Maity R K, Mazumdar A. [Distributed Newton Can Communicate Less and Resist Byzantine Workers](https://proceedings.neurips.cc/paper/2020/file/d17e6bcbcef8de3f7a00195cfa5706f1-Paper.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.
- [NIPS]Sohn J, Han D J, Choi B, et al. [Election coding for distributed learning: Protecting SignSGD against Byzantine attacks](http://proceedings.neurips.cc/paper/2020/file/a7f0d2b95c60161b3f3c82f764b1d1c9-Paper.pdf)[J]. Advances in Neural Information Processing Systems, 2020, 33.
- Guo C, Hannun A, Knott B, et al. [Secure multiparty computations in floating-point arithmetic](https://arxiv.org/pdf/2001.03192.pdf)[J]. arXiv preprint arXiv:2001.03192, 2020.
- Kemchi Sofiane, Abdelhafid Zitouni (LIRE), Mahieddine Djoudi (TECHNÉ - EA 6316) .[Self Organization Agent Oriented Dynamic Resource Allocation on Open Federated Clouds Environment](https://arxiv.org/pdf/2001.07496) [J]. arXiv preprint arXiv:2001.07496.
- Tien-Dung Cao, Tram Truong-Huu, Hien Tran, Khanh Tran .[A Federated Learning Framework for Privacy-preserving and Parallel Training](https://arxiv.org/pdf/2001.09782) [J]. arXiv preprint arXiv:2001.09782.
- Huawei Huang, Kangying Lin, Song Guo, Pan Zhou, Zibin Zheng .[Prophet: Proactive Candidate-Selection for Federated Learning by Predicting the Qualities of Training and Reporting Phases](https://arxiv.org/pdf/2002.00577) [J]. arXiv preprint arXiv:2002.00577.
- Madhusanka Manimel Wadu, Sumudu Samarakoon, Mehdi Bennis .[Federated Learning under Channel Uncertainty: Joint Client Scheduling and Resource Allocation](https://arxiv.org/pdf/2002.00802) [J]. arXiv preprint arXiv:2002.00802.
- Yingyu Li, Anqi Huang, Yong Xiao, Xiaohu Ge, Sumei Sun, Han-Chieh Chao .[Federated Orchestration for Network Slicing of Bandwidth and Computational Resource](https://arxiv.org/pdf/2002.02451) [J]. arXiv preprint arXiv:2002.02451.
- Martin Florian, Sebastian Henningsen, Björn Scheuermann .[The Sum of Its Parts: Analysis of Federated Byzantine Agreement Systems](https://arxiv.org/pdf/2002.08101) [J]. arXiv preprint arXiv:2002.08101.
- Philipp D. Rohde, Maria-Esther Vidal .[Optimizing Federated Queries Based on the Physical Design of a Data Lake](https://arxiv.org/pdf/2002.08102) [J]. arXiv preprint arXiv:2002.08102.
- Corey Tessler, Venkata P. Modekurthy, Nathan Fisher, Abusayeed Saifullah .[Bringing Inter-Thread Cache Benefits to Federated Scheduling -- Extended Results & Technical Report](https://arxiv.org/pdf/2002.12516) [J]. arXiv preprint arXiv:2002.12516.
- Yansong Gao, Minki Kim, Sharif Abuadbba, Yeonjae Kim, Chandra Thapa, Kyuyeon Kim, Seyit A. Camtepe, Hyoungshick Kim, Surya Nepal .[End-to-End Evaluation of Federated Learning and Split Learning for Internet of Things](https://arxiv.org/pdf/2003.13376) [J]. arXiv preprint arXiv:2003.13376.
- Rui Hu, Yanmin Gong, Yuanxiong Guo .[CPFed: Communication-Efficient and Privacy-Preserving Federated Learning](https://arxiv.org/pdf/2003.13761) [J]. arXiv preprint arXiv:2003.13761.
- Stefan Vlaski, Ali H. Sayed .[Second-Order Guarantees in Centralized, Federated and Decentralized Nonconvex Optimization](https://arxiv.org/pdf/2003.14366) [J]. arXiv preprint arXiv:2003.14366.
- Dale Stansberry, Suhas Somnath, Jessica Breet, Gregory Shutt, Mallikarjun Shankar .[DataFed: Towards Reproducible Research via Federated Data Management](https://arxiv.org/pdf/2004.03710) [J]. arXiv preprint arXiv:2004.03710.
- Ryan Chard, Yadu Babuji, Zhuozhao Li, Tyler Skluzacek, Anna Woodard, Ben Blaiszik, Ian Foster, Kyle Chard .[funcX: A Federated Function Serving Fabric for Science](https://arxiv.org/pdf/2005.04215) [J]. arXiv preprint arXiv:2005.04215.
- Utkarsh Chandra Srivastava, Dhruv Upadhyay, Vinayak Sharma .[Intracranial Hemorrhage Detection Using Neural Network Based Methods With Federated Learning](https://arxiv.org/pdf/2005.08644) [J]. arXiv preprint arXiv:2005.08644.
- GeunHyeong Lee, Soo-Yong Shin .[Reliability and Performance Assessment of Federated Learning on Clinical Benchmark Data](https://arxiv.org/pdf/2005.11756) [J]. arXiv preprint arXiv:2005.11756.
- Hans Albert Lianto, Yang Zhao, Jun Zhao .[Responsive Web User Interface to Recover Training Data from User Gradients in Federated Learning](https://arxiv.org/pdf/2006.04695) [J]. arXiv preprint arXiv:2006.04695.
- [NIPS]Woodworth B, Patel K K, Srebro N. [Minibatch vs Local SGD for Heterogeneous Distributed Learning](https://arxiv.org/pdf/2006.04735)[J]. arXiv preprint arXiv:2006.04735, 2020.
- Zhize Li, Peter Richtárik .[A Unified Analysis of Stochastic Gradient Methods for Nonconvex Federated Optimization](https://arxiv.org/pdf/2006.07013) [J]. arXiv preprint arXiv:2006.07013.
- Mohammad Rasouli, Tao Sun, Ram Rajagopal .[FedGAN: Federated Generative Adversarial Networks for Distributed Data](https://arxiv.org/pdf/2006.07228) [J]. arXiv preprint arXiv:2006.07228.
- Yann Fraboni, Richard Vidal, Marco Lorenzi .[Free-rider Attacks on Model Aggregation in Federated Learning](https://arxiv.org/pdf/2006.11901) [J]. arXiv preprint arXiv:2006.11901.
- Anis Elgabli, Jihong Park, Chaouki Ben Issaid, Mehdi Bennis .[Harnessing Wireless Channels for Scalable and Privacy-Preserving Federated Learning](https://arxiv.org/pdf/2007.01790) [J]. arXiv preprint arXiv:2007.01790.
- Wenchao Xia, Tony Q. S. Quek, Kun Guo, Wanli Wen, Howard H. Yang, Hongbo Zhu .[Multi-Armed Bandit Based Client Scheduling for Federated Learning](https://arxiv.org/pdf/2007.02315) [J]. arXiv preprint arXiv:2007.02315.
- Saurav Prakash, Sagar Dhakal, Mustafa Akdeniz, A. Salman Avestimehr, Nageen Himayat .[Coded Computing for Federated Learning at the Edge](https://arxiv.org/pdf/2007.03273) [J]. arXiv preprint arXiv:2007.03273.
- Zhaohui Yang, Mingzhe Chen, Walid Saad, Choong Seon Hong, Mohammad Shikh-Bahaei, H. Vincent Poor, Shuguang Cui .[Delay Minimization for Federated Learning Over Wireless Communication Networks](https://arxiv.org/pdf/2007.03462) [J]. arXiv preprint arXiv:2007.03462.
- Kun Li, Fanglan Zheng, Jiang Tian, Xiaojia Xiang .[A Federated F-score Based Ensemble Model for Automatic Rule Extraction](https://arxiv.org/pdf/2007.03533) [J]. arXiv preprint arXiv:2007.03533.
- Mustafa Safa Ozdayi, Murat Kantarcioglu, Yulia R. Gel .[Defending Against Backdoors in Federated Learning with Robust Learning Rate](https://arxiv.org/pdf/2007.03767) [J]. arXiv preprint arXiv:2007.03767.
- Yutao Huang, Lingyang Chu, Zirui Zhou, Lanjun Wang, Jiangchuan Liu, Jian Pei, Yong Zhang .[Personalized Federated Learning: An Attentive Collaboration Approach](https://arxiv.org/pdf/2007.03797) [J]. arXiv preprint arXiv:2007.03797.
- Vaikkunth Mugunthan, Ravi Rahman, Lalana Kagal .[BlockFLow: An Accountable and Privacy-Preserving Solution for Federated Learning](https://arxiv.org/pdf/2007.03856) [J]. arXiv preprint arXiv:2007.03856.
- Hossein Hosseini, Sungrack Yun, Hyunsin Park, Christos Louizos, Joseph Soriaga, Max Welling .[Federated Learning of User Authentication Models](https://arxiv.org/pdf/2007.04618) [J]. arXiv preprint arXiv:2007.04618.
- [NIPS]Hongyi Wang, Kartik Sreenivasan, Shashank Rajput, Harit Vishwakarma, Saurabh Agarwal, Jy-yong Sohn, Kangwook Lee, Dimitris Papailiopoulos .[Attack of the Tails: Yes, You Really Can Backdoor Federated Learning](https://arxiv.org/pdf/2007.05084) [J]. arXiv preprint arXiv:2007.05084.
- Mikko A. Heikkilä, Antti Koskela, Kana Shimizu, Samuel Kaski, Antti Honkela .[Differentially private cross-silo federated learning](https://arxiv.org/pdf/2007.05553) [J]. arXiv preprint arXiv:2007.05553.
- Boyi Liu, Bingjie Yan, Yize Zhou, Yifan Yang, Yixian Zhang .[Experiments of Federated Learning for COVID-19 Chest X-ray Images](https://arxiv.org/pdf/2007.05592) [J]. arXiv preprint arXiv:2007.05592.
- Zhaonan Qu, Kaixiang Lin, Jayant Kalagnanam, Zhaojian Li, Jiayu Zhou, Zhengyuan Zhou .[Federated Learning's Blessing: FedAvg has Linear Speedup](https://arxiv.org/pdf/2007.05690) [J]. arXiv preprint arXiv:2007.05690.
- Balázs Pejó .[The Good, The Bad, and The Ugly: Quality Inference in Federated Learning](https://arxiv.org/pdf/2007.06236) [J]. arXiv preprint arXiv:2007.06236.
- Jer Shyuan Ng, Wei Yang Bryan Lim, Hong-Ning Dai, Zehui Xiong, Jianqiang Huang, Dusit Niyato, Xian-Sheng Hua, Cyril Leung, Chunyan Miao .[Joint Auction-Coalition Formation Framework for Communication-Efficient Federated Learning in UAV-Enabled Internet of Vehicles](https://arxiv.org/pdf/2007.06378) [J]. arXiv preprint arXiv:2007.06378.
- Rajesh Kumar, Abdullah Aman Khan, Sinmin Zhang, WenYong Wang, Yousif Abuidris, Waqas Amin, Jay Kumar .[Blockchain-Federated-Learning and Deep Learning Models for COVID-19 detection using CT Imaging](https://arxiv.org/pdf/2007.06537) [J]. arXiv preprint arXiv:2007.06537.
- Qunsong Zeng, Yuqing Du, Kaibin Huang, Kin K. Leung .[Energy-Efficient Resource Management for Federated Edge Learning with CPU-GPU Heterogeneous Computing](https://arxiv.org/pdf/2007.07122) [J]. arXiv preprint arXiv:2007.07122.
- Wenqi Shi, Sheng Zhou, Zhisheng Niu, Miao Jiang, Lu Geng .[Joint Device Scheduling and Resource Allocation for Latency Constrained Wireless Federated Learning](https://arxiv.org/pdf/2007.07174) [J]. arXiv preprint arXiv:2007.07174.
- Hanchi Ren, Jingjing Deng, Xianghua Xie .[Privacy Preserving Text Recognition with Gradient-Boosting for Federated Learning](https://arxiv.org/pdf/2007.07296) [J]. arXiv preprint arXiv:2007.07296.
- [ICML][communication]Daniel Rothchild, Ashwinee Panda, Enayat Ullah, Nikita Ivkin, Ion Stoica, Vladimir Braverman, Joseph Gonzalez, Raman Arora .[FetchSGD: Communication-Efficient Federated Learning with Sketching](https://arxiv.org/pdf/2007.07682) [J]. arXiv preprint arXiv:2007.07682.<br>[code:[kiddyboots216/CommEfficient](https://github.com/kiddyboots216/CommEfficient); video:[fetchsgd-communicationefficient-federated-learning-with-sketching](https://slideslive.com/38928454/fetchsgd-communicationefficient-federated-learning-with-sketching)]
- Shashank Jere, Qiang Fan, Bodong Shang, Lianjun Li, Lingjia Liu .[Federated Learning in Mobile Edge Computing: An Edge-Learning Perspective for Beyond 5G](https://arxiv.org/pdf/2007.08030) [J]. arXiv preprint arXiv:2007.08030.
- Rafa Gâlvez, Veelasha Moonsamy, Claudia Diaz .[Less is More: A privacy-respecting Android malware classifier using Federated Learning](https://arxiv.org/pdf/2007.08319) [J]. arXiv preprint arXiv:2007.08319.
- Vale Tolpegin, Stacey Truex, Mehmet Emre Gursoy, Ling Liu .[Data Poisoning Attacks Against Federated Learning Systems](https://arxiv.org/pdf/2007.08432) [J]. arXiv preprint arXiv:2007.08432.
- Vito Walter Anelli, Yashar Deldjoo, Tommaso Di Noia, Antonio Ferrara .[Prioritized Multi-Criteria Federated Learning](https://arxiv.org/pdf/2007.08893) [J]. arXiv preprint arXiv:2007.08893.
- Marten van Dijk, Nhuong V. Nguyen, Toan N. Nguyen, Lam M. Nguyen, Quoc Tran-Dinh, Phuong Ha Nguyen .[Asynchronous Federated Learning with Reduced Number of Rounds and with Differential Privacy from Less Aggregated Gaussian Noise](https://arxiv.org/pdf/2007.09208) [J]. arXiv preprint arXiv:2007.09208.
- Jed Mills, Jia Hu, Geyong Min .[User-Oriented Multi-Task Federated Deep Learning for Mobile Edge Computing](https://arxiv.org/pdf/2007.09236) [J]. arXiv preprint arXiv:2007.09236.
- Seyyedali Hosseinalipour, Sheikh Shams Azam, Christopher G. Brinton, Nicolo Michelusi, Vaneet Aggarwal, David J. Love, Huaiyu Dai .[Multi-Stage Hybrid Federated Learning over Large-Scale Wireless Fog Networks](https://arxiv.org/pdf/2007.09511) [J]. arXiv preprint arXiv:2007.09511.
- Yi Liu, Sahil Garg, Jiangtian Nie, Yang Zhang, Zehui Xiong, Jiawen Kang, M. Shamim Hossain .[Deep Anomaly Detection for Time-series Data in Industrial IoT: A Communication-Efficient On-device Federated Learning Approach](https://arxiv.org/pdf/2007.09712) [J]. arXiv preprint arXiv:2007.09712.
- Zhaoxiong Yang, Shuihai Hu, Kai Chen .[FPGA-Based Hardware Accelerator of Homomorphic Encryption for Efficient Federated Learning](https://arxiv.org/pdf/2007.10560) [J]. arXiv preprint arXiv:2007.10560.
- Yang Liu, Jiaheng Wei .[Incentives for Federated Learning: a Hypothesis Elicitation Approach](https://arxiv.org/pdf/2007.10596) [J]. arXiv preprint arXiv:2007.10596.
- Heiko Ludwig, Nathalie Baracaldo, Gegi Thomas, Yi Zhou, Ali Anwar, Shashank Rajamoni, Yuya Ong, Jayaram Radhakrishnan, Ashish Verma, Mathieu Sinn, Mark Purcell, Ambrish Rawat, Tran Minh, Naoise Holohan, Supriyo Chakraborty, Shalisha Whitherspoon, Dean Steuer, Laura Wynter, Hifaz Hassan, Sean Laguna, Mikhail Yurochkin, Mayank Agarwal, Ebube Chuba, Annie Abay .[IBM Federated Learning: an Enterprise Framework White Paper V0.1](https://arxiv.org/pdf/2007.10987) [J]. arXiv preprint arXiv:2007.10987.
- Jinhyun So, Basak Guler, A. Salman Avestimehr .[Byzantine-Resilient Secure Federated Learning](https://arxiv.org/pdf/2007.11115) [J]. arXiv preprint arXiv:2007.11115.
- Sin Kit Lo, Qinghua Lu, Chen Wang, Hye-Young Paik, Liming Zhu .[A Systematic Literature Review on Federated Machine Learning: From A Software Engineering Perspective](https://arxiv.org/pdf/2007.11354) [J]. arXiv preprint arXiv:2007.11354.
- Wenqing Zhang, Yang Qiu, Song Bai, Rui Zhang, Xiaolin Wei, Xiang Bai .[FedOCR: Communication-Efficient Federated Learning for Scene Text Recognition](https://arxiv.org/pdf/2007.11462) [J]. arXiv preprint arXiv:2007.11462.
- Yi Liu, Jiangtian Nie, Xuandi Li, Syed Hassan Ahmed, Wei Yang Bryan Lim, Chunyan Miao .[Federated Learning in the Sky: Aerial-Ground Air Quality Sensing Framework with UAV Swarms](https://arxiv.org/pdf/2007.12004) [J]. arXiv preprint arXiv:2007.12004.
- Chuhan Wu, Fangzhao Wu, Tao Di, Yongfeng Huang, Xing Xie .[FedCTR: Federated Native Ad CTR Prediction with Multi-Platform User Behavior Data](https://arxiv.org/pdf/2007.12135) [J]. arXiv preprint arXiv:2007.12135.
- Aaqib Saeed, Flora D. Salim, Tanir Ozcelebi, Johan Lukkien .[Federated Self-Supervised Learning of Multi-Sensor Representations for Embedded Intelligence](https://arxiv.org/pdf/2007.13018) [J]. arXiv preprint arXiv:2007.13018.
- Yuben Qu, Chao Dong, Jianchao Zheng, Qihui Wu, Yun Shen, Fan Wu, Alagan Anpalagan .[Empowering the Edge Intelligence by Air-Ground Integrated Federated Learning in 6G Networks](https://arxiv.org/pdf/2007.13054) [J]. arXiv preprint arXiv:2007.13054.
- Hung T. Nguyen, Vikash Sehwag, Seyyedali Hosseinalipour, Christopher G. Brinton, Mung Chiang, H. Vincent Poor .[Fast-Convergent Federated Learning](https://arxiv.org/pdf/2007.13137) [J]. arXiv preprint arXiv:2007.13137.
- Chandra Thapa, Jun Wen Tang, Sharif Abuadbba, Yansong Gao, Yifeng Zheng, Seyit A. Camtepe, Surya Nepal, Mahathir Almashor .[FedEmail: Performance Measurement of Privacy-friendly Phishing Detection Enabled by Federated Learning](https://arxiv.org/pdf/2007.13300) [J]. arXiv preprint arXiv:2007.13300.
- Anmin Fu, Xianglong Zhang, Naixue Xiong, Yansong Gao, Huaqun Wang .[VFL: A Verifiable Federated Learning with Privacy-Preserving for Big Data in Industrial IoT](https://arxiv.org/pdf/2007.13585) [J]. arXiv preprint arXiv:2007.13585.
- Yue Xiao, Yu Ye, Shaocheng Huang, Li Hao, Zheng Ma, Ming Xiao, Shahid Mumtaz .[Fully Decentralized Federated Learning Based Beamforming Design for UAV Communications](https://arxiv.org/pdf/2007.13614) [J]. arXiv preprint arXiv:2007.13614.
- Wentai Wu, Ligang He, Weiwei Lin, Rui Mao .[Accelerating Federated Learning over Reliability-Agnostic Clients in Mobile Edge Computing Systems](https://arxiv.org/pdf/2007.14374) [J]. arXiv preprint arXiv:2007.14374.
- Constance Beguier, Eric W. Tramel .[SAFER: Sparse Secure Aggregation for Federated Learning](https://arxiv.org/pdf/2007.14861) [J]. arXiv preprint arXiv:2007.14861.
- Ruichen Jiang, Sheng Zhou .[Cluster-Based Cooperative Digital Over-the-Air Aggregation for Wireless Federated Edge Learning](https://arxiv.org/pdf/2008.00994) [J]. arXiv preprint arXiv:2008.00994.
- Rui Hu, Yanmin Gong, Yuanxiong Guo .[Sparsified Privacy-Masking for Communication-Efficient and Privacy-Preserving Federated Learning](https://arxiv.org/pdf/2008.01558) [J]. arXiv preprint arXiv:2008.01558.
- Dimitrios Dimitriadis, Kenichi Kumatani, Robert Gmyr, Yashesh Gaur, Sefik Emre Eskimez .[Federated Transfer Learning with Dynamic Gradient Aggregation](https://arxiv.org/pdf/2008.02452) [J]. arXiv preprint arXiv:2008.02452.
- Huafei Zhu .[On the relationship between (secure) multi-party computation and (secure) federated learning](https://arxiv.org/pdf/2008.02609) [J]. arXiv preprint arXiv:2008.02609.
- Filip Granqvist, Matt Seigel, Rogier van Dalen, Áine Cahill, Stephen Shum, Matthias Paulik .[Improving on-device speaker verification using federated learning with privacy](https://arxiv.org/pdf/2008.02651) [J]. arXiv preprint arXiv:2008.02651.
- Ang Li, Jingwei Sun, Binghui Wang, Lin Duan, Sicheng Li, Yiran Chen, Hai Li .[LotteryFL: Personalized and Communication-Efficient Federated Learning with Lottery Ticket Hypothesis on Non-IID Datasets](https://arxiv.org/pdf/2008.03371) [J]. arXiv preprint arXiv:2008.03371.
- Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, Ananda Theertha Suresh .[Mime: Mimicking Centralized Stochastic Algorithms in Federated Learning](https://arxiv.org/pdf/2008.03606) [J]. arXiv preprint arXiv:2008.03606.
- Jack Goetz, Ambuj Tewari .[Federated Learning via Synthetic Data](https://arxiv.org/pdf/2008.04489) [J]. arXiv preprint arXiv:2008.04489.
- Kenta Nagura, Song Bian, Takashi Sato .[FedNNNN: Norm-Normalized Neural Network Aggregation for Fast and Accurate Federated Learning](https://arxiv.org/pdf/2008.04538) [J]. arXiv preprint arXiv:2008.04538.
- Shahar Azulay, Lior Raz, Amir Globerson, Tomer Koren, Yehuda Afek .[Holdout SGD: Byzantine Tolerant Federated Learning](https://arxiv.org/pdf/2008.04612) [J]. arXiv preprint arXiv:2008.04612.
- Jiawen Kang, Zehui Xiong, Chunxiao Jiang, Yi Liu, Song Guo, Yang Zhang, Dusit Niyato, Cyril Leung, Chunyan Miao .[Scalable and Communication-efficient Decentralized Federated Edge Learning with Multi-blockchain Framework](https://arxiv.org/pdf/2008.04743) [J]. arXiv preprint arXiv:2008.04743.
- Farzin Haddadpour, Belhal Karimi, Ping Li, Xiaoyun Li .[FedSKETCH: Communication-Efficient and Private Federated Learning via Sketching](https://arxiv.org/pdf/2008.04975) [J]. arXiv preprint arXiv:2008.04975.
- Dianbo Sui, Yubo Chen, Kang Liu, Jun Zhao .[Distantly Supervised Relation Extraction in Federated Settings](https://arxiv.org/pdf/2008.05049) [J]. arXiv preprint arXiv:2008.05049.
- Latif U. Khan, Walid Saad, Zhu Han, Choong Seon Hong .[Dispersed Federated Learning: Vision, Taxonomy, and Future Directions](https://arxiv.org/pdf/2008.05189) [J]. arXiv preprint arXiv:2008.05189.
- Weituo Hao, Nikhil Mehta, Kevin J Liang, Pengyu Cheng, Mostafa El-Khamy, Lawrence Carin .[WAFFLe: Weight Anonymized Factorization for Federated Learning](https://arxiv.org/pdf/2008.05687) [J]. arXiv preprint arXiv:2008.05687.
- Yuncheng Wu, Shaofeng Cai, Xiaokui Xiao, Gang Chen, Beng Chin Ooi .[Privacy Preserving Vertical Federated Learning for Tree-based Models](https://arxiv.org/pdf/2008.06170) [J]. arXiv preprint arXiv:2008.06170.
- Sohei Itahara, Takayuki Nishio, Yusuke Koda, Masahiro Morikura, Koji Yamamoto .[Distillation-Based Semi-Supervised Federated Learning for Communication-Efficient Collaborative Training with Non-IID Private Data](https://arxiv.org/pdf/2008.06180) [J]. arXiv preprint arXiv:2008.06180.
- Bin Gu, Zhiyuan Dang, Xiang Li, Heng Huang .[Federated Doubly Stochastic Kernel Learning for Vertically Partitioned Data](https://arxiv.org/pdf/2008.06197) [J]. arXiv preprint arXiv:2008.06197.
- Lixu Wang, Shichao Xu, Xiao Wang, Qi Zhu .[Towards Class Imbalance in Federated Learning](https://arxiv.org/pdf/2008.06217) [J]. arXiv preprint arXiv:2008.06217.
- Bin Gu, An Xu, Zhouyuan Huo, Cheng Deng, Heng Huang .[Privacy-Preserving Asynchronous Federated Learning Algorithms for Multi-Party Vertically Collaborative Learning](https://arxiv.org/pdf/2008.06233) [J]. arXiv preprint arXiv:2008.06233.
- Mingshu Cong, Han Yu, Xi Weng, Jiabao Qu, Yang Liu, Siu Ming Yiu .[A VCG-based Fair Incentive Mechanism for Federated Learning](https://arxiv.org/pdf/2008.06680) [J]. arXiv preprint arXiv:2008.06680.
- Fuxun Yu, Weishan Zhang, Zhuwei Qin, Zirui Xu, Di Wang, Chenchen Liu, Zhi Tian, Xiang Chen .[Heterogeneous Federated Learning](https://arxiv.org/pdf/2008.06767) [J]. arXiv preprint arXiv:2008.06767.
- Antonious M. Girgis, Deepesh Data, Suhas Diggavi, Peter Kairouz, Ananda Theertha Suresh .[Shuffled Model of Federated Learning: Privacy, Communication and Accuracy Trade-offs](https://arxiv.org/pdf/2008.07180) [J]. arXiv preprint arXiv:2008.07180.
- Vito Walter Anelli, Yashar Deldjoo, Tommaso Di Noia, Antonio Ferrara, Fedelucio Narducci .[How to Put Users in Control of their Data via Federated Pair-Wise Recommendation](https://arxiv.org/pdf/2008.07192) [J]. arXiv preprint arXiv:2008.07192.
- Yuan Liang, Yange Guo, Yanxia Gong, Chunjie Luo, Jianfeng Zhan, Yunyou Huang .[An Isolated Data Island Benchmark Suite for Federated Learning](https://arxiv.org/pdf/2008.07257) [J]. arXiv preprint arXiv:2008.07257.
- Buse Gul Atli, Yuxi Xia, Samuel Marchal, N. Asokan .[WAFFLE: Watermarking in Federated Learning](https://arxiv.org/pdf/2008.07298) [J]. arXiv preprint arXiv:2008.07298.
- Mathieu Andreux, Jean Ogier du Terrail, Constance Beguier, Eric W. Tramel .[Siloed Federated Learning for Multi-Centric Histopathology Datasets](https://arxiv.org/pdf/2008.07424) [J]. arXiv preprint arXiv:2008.07424.
- Minchul Kim, Jungwoo Lee .[Information-Theoretic Privacy in Federated Submodel learning](https://arxiv.org/pdf/2008.07656) [J]. arXiv preprint arXiv:2008.07656.
- [MICCAIW]Yousef Yeganeh, Azade Farshad, Nassir Navab, Shadi Albarqouni .[Inverse Distance Aggregation for Federated Learning with Non-IID Data](https://arxiv.org/pdf/2008.07665) [J]. arXiv preprint arXiv:2008.07665.
- Junjie Tan, Ying-Chang Liang, Nguyen Cong Luong, Dusit Niyato .[Toward Smart Security Enhancement of Federated Learning Networks](https://arxiv.org/pdf/2008.08330) [J]. arXiv preprint arXiv:2008.08330.
- Frank Po-Chen Lin, Christopher G. Brinton, Nicolò Michelusi .[Federated Learning with Communication Delay in Edge Networks](https://arxiv.org/pdf/2008.09323) [J]. arXiv preprint arXiv:2008.09323.
- Behzad Khamidehi, Elvino S. Sousa .[Federated Learning for Cellular-connected UAVs: Radio Mapping and Path Planning](https://arxiv.org/pdf/2008.10054) [J]. arXiv preprint arXiv:2008.10054.
- Mingkai Huang, Hao Li, Bing Bai, Chang Wang, Kun Bai, Fei Wang .[A Federated Multi-View Deep Learning Framework for Privacy-Preserving Recommendations](https://arxiv.org/pdf/2008.10808) [J]. arXiv preprint arXiv:2008.10808.
- Yan Kang, Yang Liu, Tianjian Chen .[FedMVT: Semi-supervised Vertical Federated Learning with MultiView Training](https://arxiv.org/pdf/2008.10838) [J]. arXiv preprint arXiv:2008.10838.
- Ahmet M. Elbir, Sinem Coleri .[Federated Learning for Channel Estimation in Conventional and IRS-Assisted Massive MIMO](https://arxiv.org/pdf/2008.10846) [J]. arXiv preprint arXiv:2008.10846.
- Mohammad Mohammadi Amiri, Deniz Gunduz, Sanjeev R. Kulkarni, H. Vincent Poor .[Convergence of Federated Learning over a Noisy Downlink](https://arxiv.org/pdf/2008.11141) [J]. arXiv preprint arXiv:2008.11141.
- Dimitris Stripelis, Jose Luis Ambite .[Accelerating Federated Learning in Heterogeneous Data and Computational Environments](https://arxiv.org/pdf/2008.11281) [J]. arXiv preprint arXiv:2008.11281.
- Zhengming Zhang, Zhewei Yao, Yaoqing Yang, Yujun Yan, Joseph E. Gonzalez, Michael W. Mahoney .[Benchmarking Semi-supervised Federated Learning](https://arxiv.org/pdf/2008.11364) [J]. arXiv preprint arXiv:2008.11364.
- Dongming Han, Wei Chen, Rusheng Pan, Yijing Liu, Jiehui Zhou, Ying Xu, Tianye Zhang, Changjie Fan, Jianrong Tao, Xiaolong (Luke)Zhang .[GraphFederator: Federated Visual Analysis for Multi-party Graphs](https://arxiv.org/pdf/2008.11989) [J]. arXiv preprint arXiv:2008.11989.
- Lingjuan Lyu, Xinyi Xu, Qian Wang .[Collaborative Fairness in Federated Learning](https://arxiv.org/pdf/2008.12161) [J]. arXiv preprint arXiv:2008.12161.
- Tejaswini Mallavarapu, Luke Cranfill, Junggab Son, Eun Hye Kim, Reza M. Parizi, John Morris .[A Federated Approach for Fine-Grained Classification of Fashion Apparel](https://arxiv.org/pdf/2008.12350) [J]. arXiv preprint arXiv:2008.12350.
- Seok-Ju Hahn, Junghye Lee .[GRAFFL: Gradient-free Federated Learning of a Bayesian Generative Model](https://arxiv.org/pdf/2008.12925) [J]. arXiv preprint arXiv:2008.12925.
- Afaf Taïk, Soumaya Cherkaoui .[Federated Edge Learning : Design Issues and Challenges](https://arxiv.org/pdf/2009.00081) [J]. arXiv preprint arXiv:2009.00081.
- Sinem Sav, Apostolos Pyrgelis, Juan R. Troncoso-Pastoriza, David Froelicher, Jean-Philippe Bossuat, Joao Sa Sousa, Jean-Pierre Hubaux .[POSEIDON: Privacy-Preserving Federated Neural Network Learning](https://arxiv.org/pdf/2009.00349) [J]. arXiv preprint arXiv:2009.00349.
- Daiqing Li, Amlan Kar, Nishant Ravikumar, Alejandro F Frangi, Sanja Fidler .[Fed-Sim: Federated Simulation for Medical Imaging](https://arxiv.org/pdf/2009.00668) [J]. arXiv preprint arXiv:2009.00668.
- Sheng Lin, Chenghong Wang, Hongjia Li, Jieren Deng, Yanzhi Wang, Caiwen Ding .[ESMFL: Efficient and Secure Models for Federated Learning](https://arxiv.org/pdf/2009.01867) [J]. arXiv preprint arXiv:2009.01867.
- Holger R. Roth, Ken Chang, Praveer Singh, Nir Neumark, Wenqi Li, Vikash Gupta, Sharut Gupta, Liangqiong Qu, Alvin Ihsani, Bernardo C. Bizzo, Yuhong Wen, Varun Buch, Meesam Shah, Felipe Kitamura, Matheus Mendonça, Vitor Lavor, Ahmed Harouni, Colin Compas, Jesse Tetreault, Prerna Dogra, Yan Cheng, Selnur Erdal, Richard White, Behrooz Hashemian, Thomas Schultz, Miao Zhang, Adam McCarthy, B. Min Yun, Elshaimaa Sharaf, Katharina V. Hoebel, Jay B. Patel, Bryan Chen, Sean Ko, Evan Leibovitz, Etta D. Pisano, Laura Coombs, Daguang Xu, Keith J. Dreyer, Ittai Dayan, Ram C. Naidu, Mona Flores, Daniel Rubin, Jayashree Kalpathy-Cramer .[Federated Learning for Breast Density Classification: A Real-World Implementation](https://arxiv.org/pdf/2009.01871) [J]. arXiv preprint arXiv:2009.01871.
- Hong-You Chen, Wei-Lun Chao .[FedDistill: Making Bayesian Model Ensemble Applicable to Federated Learning](https://arxiv.org/pdf/2009.01974) [J]. arXiv preprint arXiv:2009.01974.
- Tung T. Vu, Duy T. Ngo, Hien Quoc Ngo, Minh N. Dao, Nguyen H. Tran, Richard H. Middleton .[User Selection Approaches to Mitigate the Straggler Effect for Federated Learning on Cell-Free Massive MIMO Networks](https://arxiv.org/pdf/2009.02031) [J]. arXiv preprint arXiv:2009.02031.
- Pei Fang, Zhendong Cai, Hui Chen, QingJiang Shi .[FLFE: A Communication-Efficient and Privacy-Preserving Federated Feature Engineering Framework](https://arxiv.org/pdf/2009.02557) [J]. arXiv preprint arXiv:2009.02557.
- Basheer Qolomany, Kashif Ahmad, Ala Al-Fuqaha, Junaid Qadir .[Particle Swarm Optimized Federated Learning For Industrial IoT and Smart City Services](https://arxiv.org/pdf/2009.02560) [J]. arXiv preprint arXiv:2009.02560.
- Weishan Zhang, Qinghua Lu, Qiuyu Yu, Zhaotong Li, Yue Liu, Sin Kit Lo, Shiping Chen, Xiwei Xu, Liming Zhu .[Blockchain-based Federated Learning for Failure Detection in Industrial IoT](https://arxiv.org/pdf/2009.02643) [J]. arXiv preprint arXiv:2009.02643.
- Chang Wang, Jian Liang, Mingkai Huang, Bing Bai, Kun Bai, Hao Li .[Hybrid Differentially Private Federated Learning on Vertically Partitioned Data](https://arxiv.org/pdf/2009.02763) [J]. arXiv preprint arXiv:2009.02763.
- Boyi Liu, Bingjie Yan, Yize Zhou, Jun Wang, Li Liu, Yuhan Zhang, Xiaolan Nie .[A Real-time Contribution Measurement Method for Participants in Federated Learning](https://arxiv.org/pdf/2009.03510) [J]. arXiv preprint arXiv:2009.03510.
- Mohammad Naseri, Jamie Hayes, Emiliano De Cristofaro .[Toward Robustness and Privacy in Federated Learning: Experimenting with Local and Central Differential Privacy](https://arxiv.org/pdf/2009.03561) [J]. arXiv preprint arXiv:2009.03561.
- Maria Peifer, Alejandro Ribeiro .[Federated Classification using Parsimonious Functions in Reproducing Kernel Hilbert Spaces](https://arxiv.org/pdf/2009.03768) [J]. arXiv preprint arXiv:2009.03768.
- Lichao Sun, Lingjuan Lyu .[Federated Model Distillation with Noise-Free Differential Privacy](https://arxiv.org/pdf/2009.05537) [J]. arXiv preprint arXiv:2009.05537.
- Rui Hu, Yanmin Gong .[Trading Data For Learning: Incentive Mechanism For On-Device Federated Learning](https://arxiv.org/pdf/2009.05604) [J]. arXiv preprint arXiv:2009.05604.
- Sudipta Paul, Poushali Sengupta, Subhankar Mishra .[FLaPS: Federated Learning and Privately Scaling](https://arxiv.org/pdf/2009.06005) [J]. arXiv preprint arXiv:2009.06005.
- Tianhao Wang, Johannes Rausch, Ce Zhang, Ruoxi Jia, Dawn Song .[A Principled Approach to Data Valuation for Federated Learning](https://arxiv.org/pdf/2009.06192) [J]. arXiv preprint arXiv:2009.06192.
- Fanglan Zheng, Erihe, Kun Li, Jiang Tian, Xiaojia Xiang .[A Vertical Federated Learning Method for Interpretable Scorecard and Its Application in Credit Scoring](https://arxiv.org/pdf/2009.06218) [J]. arXiv preprint arXiv:2009.06218.
- Pengqian Yu, Laura Wynter, Shiau Hong Lim .[Fed+: A Family of Fusion Algorithms for Federated Learning](https://arxiv.org/pdf/2009.06303) [J]. arXiv preprint arXiv:2009.06303.
- Rahif Kassab, Osvaldo Simeone .[Federated Generalized Bayesian Learning via Distributed Stein Variational Gradient Descent](https://arxiv.org/pdf/2009.06419) [J]. arXiv preprint arXiv:2009.06419.
- Qianqian Tong, Guannan Liang, Jinbo Bi .[Effective Federated Adaptive Gradient Methods with Non-IID Decentralized Data](https://arxiv.org/pdf/2009.06557) [J]. arXiv preprint arXiv:2009.06557.
- Anxun He, Jianzong Wang, Zhangcheng Huang, Jing Xiao .[FedSmart: An Auto Updating Federated Learning Optimization Mechanism](https://arxiv.org/pdf/2009.07455) [J]. arXiv preprint arXiv:2009.07455.
- Yanlin Zhou, George Pu, Xiyao Ma, Xiaolin Li, Dapeng Wu .[Distilled One-Shot Federated Learning](https://arxiv.org/pdf/2009.07999) [J]. arXiv preprint arXiv:2009.07999.
- Ruixuan Liu, Yang Cao, Hong Chen, Ruoyang Guo, Masatoshi Yoshikawa .[FLAME: Differentially Private Federated Learning in the Shuffle Model](https://arxiv.org/pdf/2009.08063) [J]. arXiv preprint arXiv:2009.08063.
- Jie Peng, Zhaoxian Wu, Qing Ling .[Byzantine-Robust Variance-Reduced Federated Learning over Distributed Non-i.i.d. Data](https://arxiv.org/pdf/2009.08161) [J]. arXiv preprint arXiv:2009.08161.
- Matei Grama, Maria Musat, Luis Muñoz-González, Jonathan Passerat-Palmbach, Daniel Rueckert, Amir Alansary .[Robust Aggregation for Adaptive Privacy Preserving Federated Learning in Healthcare](https://arxiv.org/pdf/2009.08294) [J]. arXiv preprint arXiv:2009.08294.
- Zhengjie Yang, Wei Bao, Dong Yuan, Nguyen H. Tran, Albert Y. Zomaya .[Federated Learning with Nesterov Accelerated Gradient Momentum Method](https://arxiv.org/pdf/2009.08716) [J]. arXiv preprint arXiv:2009.08716.
- Chuan Ma, Jun Li, Ming Ding, Long Shi, Taotao Wang, Zhu Han, H. Vincent Poor .[When Federated Learning Meets Blockchain: A New Distributed Learning Paradigm](https://arxiv.org/pdf/2009.09338) [J]. arXiv preprint arXiv:2009.09338.
- Takayuki Nishio, Ryoichi Shinkuma, Narayan B. Mandayam .[Estimation of Individual Device Contributions for Incentivizing Federated Learning](https://arxiv.org/pdf/2009.09371) [J]. arXiv preprint arXiv:2009.09371.
- Javad Mohammadi, Jesse Thornburg .[Connecting Distributed Pockets of EnergyFlexibility through Federated Computations:Limitations and Possibilities](https://arxiv.org/pdf/2009.10182) [J]. arXiv preprint arXiv:2009.10182.
- Ming Y. Lu, Dehan Kong, Jana Lipkova, Richard J. Chen, Rajendra Singh, Drew F. K. Williamson, Tiffany Y. Chen, Faisal Mahmood .[Federated Learning for Computational Pathology on Gigapixel Whole Slide Images](https://arxiv.org/pdf/2009.10190) [J]. arXiv preprint arXiv:2009.10190.
- Tra Huong Thi Le, Nguyen H. Tran, Yan Kyaw Tun, Minh N. H. Nguyen, Shashi Raj Pandey, Zhu Han, Choong Seon Hong .[An Incentive Mechanism for Federated Learning in Wireless Cellular network: An Auction Approach](https://arxiv.org/pdf/2009.10269) [J]. arXiv preprint arXiv:2009.10269.
- Weishan Zhang, Tao Zhou, Qinghua Lu, Xiao Wang, Chunsheng Zhu, Haoyun Sun, Zhipeng Wang, Sin Kit Lo, Fei-Yue Wang .[Dynamic Fusion based Federated Learning for COVID-19 Detection](https://arxiv.org/pdf/2009.10401) [J]. arXiv preprint arXiv:2009.10401.
- Shuai Yu, Xu Chen, Zhi Zhou, Xiaowen Gong, Di Wu .[When Deep Reinforcement Learning Meets Federated Learning: Intelligent Multi-Timescale Resource Management for Multi-access Edge Computing in 5G Ultra Dense Network](https://arxiv.org/pdf/2009.10601) [J]. arXiv preprint arXiv:2009.10601.
- Cheng Chen, Ziyi Chen, Yi Zhou, Bhavya Kailkhura .[FedCluster: Boosting the Convergence of Federated Learning via Cluster-Cycling](https://arxiv.org/pdf/2009.10748) [J]. arXiv preprint arXiv:2009.10748.
- Zhuoran Ma, Jianfeng Ma, Yinbin Miao, Ximeng Liu, Kim-Kwang Raymond Choo, Robert H. Deng .[Pocket Diagnosis: Secure Federated Learning against Poisoning Attack in the Cloud](https://arxiv.org/pdf/2009.10918) [J]. arXiv preprint arXiv:2009.10918.
- Swanand Kadhe, Nived Rajaraman, O. Ozan Koyluoglu, Kannan Ramchandran .[FastSecAgg: Scalable Secure Aggregation for Privacy-Preserving Federated Learning](https://arxiv.org/pdf/2009.11248) [J]. arXiv preprint arXiv:2009.11248.
- Tomer Sery, Nir Shlezinger, Kobi Cohen, Yonina C. Eldar .[Over-the-Air Federated Learning from Heterogeneous Data](https://arxiv.org/pdf/2009.12787) [J]. arXiv preprint arXiv:2009.12787.
- Shaoming Song, Yunfeng Shao, Jian Li .[Loosely Coupled Federated Learning Over Generative Models](https://arxiv.org/pdf/2009.12999) [J]. arXiv preprint arXiv:2009.12999.
- Latif U. Khan, Walid Saad, Zhu Han, Ekram Hossain, Choong Seon Hong .[Federated Learning for Internet of Things: Recent Advances, Taxonomy, and Open Challenges](https://arxiv.org/pdf/2009.13012) [J]. arXiv preprint arXiv:2009.13012.
- Naoya Yoshida, Takayuki Nishio, Masahiro Morikura, Koji Yamamoto .[MAB-based Client Selection for Federated Learning with Uncertain Resources in Mobile Networks](https://arxiv.org/pdf/2009.13879) [J]. arXiv preprint arXiv:2009.13879.
- Ahmed Roushdy Elkordy, A. Salman Avestimehr .[Secure Aggregation with Heterogeneous Quantization in Federated Learning](https://arxiv.org/pdf/2009.14388) [J]. arXiv preprint arXiv:2009.14388.
- Laércio Lima Pilla (ParSys - LRI) .[Optimal Task Assignment to Heterogeneous Federated Learning Devices](https://arxiv.org/pdf/2010.00239) [J]. arXiv preprint arXiv:2010.00239.
- Kate Donahue, Jon Kleinberg .[Model-sharing Games: Analyzing Federated Learning Under Voluntary Participation](https://arxiv.org/pdf/2010.00753) [J]. arXiv preprint arXiv:2010.00753.
- Qinbin Li, Bingsheng He, Dawn Song .[Model-Agnostic Round-Optimal Federated Learning via Knowledge Transfer](https://arxiv.org/pdf/2010.01017) [J]. arXiv preprint arXiv:2010.01017.
- Zhuqing Jia, Syed A. Jafar .[$X$-Secure $T$-Private Federated Submodel Learning](https://arxiv.org/pdf/2010.01059) [J]. arXiv preprint arXiv:2010.01059.
- Yae Jee Cho, Jianyu Wang, Gauri Joshi .[Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies](https://arxiv.org/pdf/2010.01243) [J]. arXiv preprint arXiv:2010.01243.
- Enmao Diao, Jie Ding, Vahid Tarokh .[HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients](https://arxiv.org/pdf/2010.01264) [J]. arXiv preprint arXiv:2010.01264.
- Edvin Listo Zec, Olof Mogren, John Martinsson, Leon René Sütfeld, Daniel Gillblad .[Federated learning using a mixture of experts](https://arxiv.org/pdf/2010.02056) [J]. arXiv preprint arXiv:2010.02056.
- [NIPS][non-I.I.D,personalization]Filip Hanzely, Slavomír Hanzely, Samuel Horváth, Peter Richtárik .[Lower Bounds and Optimal Algorithms for Personalized Federated Learning](https://arxiv.org/pdf/2010.02372) [J]. arXiv preprint arXiv:2010.02372.
- Alyazeed Albasyoni, Mher Safaryan, Laurent Condat, Peter Richtárik .[Optimal Gradient Compression for Distributed and Federated Learning](https://arxiv.org/pdf/2010.03246) [J]. arXiv preprint arXiv:2010.03246.
- Yuqing Zhu, Xiang Yu, Yi-Hsuan Tsai, Francesco Pittaluga, Masoud Faraki, Manmohan chandraker, Yu-Xiang Wang .[Voting-based Approaches For Differentially Private Federated Learning](https://arxiv.org/pdf/2010.04851) [J]. arXiv preprint arXiv:2010.04851.
- Wei Du, Depeng Xu, Xintao Wu, Hanghang Tong .[Fairness-aware Agnostic Federated Learning](https://arxiv.org/pdf/2010.05057) [J]. arXiv preprint arXiv:2010.05057.
- Maruan Al-Shedivat, Jennifer Gillenwater, Eric Xing, Afshin Rostamizadeh .[Federated Learning via Posterior Averaging: A New Perspective and Practical Algorithms](https://arxiv.org/pdf/2010.05273) [J]. arXiv preprint arXiv:2010.05273.
- David Byrd, Antigoni Polychroniadou .[Differentially Private Secure Multi-Party Computation for Federated Learning in Financial Applications](https://arxiv.org/pdf/2010.05867) [J]. arXiv preprint arXiv:2010.05867.
- Zheng Chai, Yujing Chen, Liang Zhao, Yue Cheng, Huzefa Rangwala .[FedAT: A Communication-Efficient Federated Learning Method with Asynchronous Tiers under Non-IID Data](https://arxiv.org/pdf/2010.05958) [J]. arXiv preprint arXiv:2010.05958.
- Fan Lai, Xiangfeng Zhu, Harsha V. Madhyastha, Mosharaf Chowdhury .[Oort: Informed Participant Selection for Scalable Federated Learning](https://arxiv.org/pdf/2010.06081) [J]. arXiv preprint arXiv:2010.06081.
- Anwaar Ulhaq, Oliver Burmeister .[COVID-19 Imaging Data Privacy by Federated Learning Design: A Theoretical Framework](https://arxiv.org/pdf/2010.06177) [J]. arXiv preprint arXiv:2010.06177.
- Xinchi Qiu, Titouan Parcolle, Daniel J. Beutel, Taner Topal, Akhil Mathur, Nicholas D. Lane .[A first look into the carbon footprint of federated learning](https://arxiv.org/pdf/2010.06537) [J]. arXiv preprint arXiv:2010.06537.
- Moming Duan, Duo Liu, Xinyuan Ji, Renping Liu, Liang Liang, Xianzhang Chen, Yujuan Tan .[FedGroup: Ternary Cosine Similarity-based Clustered Federated Learning Framework toward High Accuracy in Heterogeneous Data](https://arxiv.org/pdf/2010.06870) [J]. arXiv preprint arXiv:2010.06870.
- Harsh Bimal Desai, Mustafa Safa Ozdayi, Murat Kantarcioglu .[BlockFLA: Accountable Federated Learning via Hybrid Blockchain Architecture](https://arxiv.org/pdf/2010.07427) [J]. arXiv preprint arXiv:2010.07427.
- Saurav Prakash, Amir Salman Avestimehr .[Mitigating Byzantine Attacks in Federated Learning](https://arxiv.org/pdf/2010.07541) [J]. arXiv preprint arXiv:2010.07541.
- Raouf Kerkouche, Gergely Ács, Claude Castelluccia .[Federated Learning in Adversarial Settings](https://arxiv.org/pdf/2010.07808) [J]. arXiv preprint arXiv:2010.07808.
- Nour Moustafa, Marwa Keshk, Essam Debie, Helge Janicke .[Federated TON_IoT Windows Datasets for Evaluating AI-based Security Applications](https://arxiv.org/pdf/2010.08522) [J]. arXiv preprint arXiv:2010.08522.
- Nathalie Majcherczyk, Nishan Srishankar, Carlo Pinciroli .[Flow-FL: Data-Driven Federated Learning for Spatio-Temporal Predictions in Multi-Robot Systems](https://arxiv.org/pdf/2010.08595) [J]. arXiv preprint arXiv:2010.08595.
- Jiale Guo, Ziyao Liu, Kwok-Yan Lam, Jun Zhao, Yiqiang Chen, Chaoping Xing .[Secure Weighted Aggregation in Federated Learning](https://arxiv.org/pdf/2010.08730) [J]. arXiv preprint arXiv:2010.08730.
- Fan Mo, Anastasia Borovykh, Mohammad Malekzadeh, Hamed Haddadi, Soteris Demetriou .[Layer-wise Characterization of Latent Information Leakage in Federated Learning](https://arxiv.org/pdf/2010.08762) [J]. arXiv preprint arXiv:2010.08762.
- Fengda Zhang, Kun Kuang, Zhaoyang You, Tao Shen, Jun Xiao, Yin Zhang, Chao Wu, Yueting Zhuang, Xiaolin Li .[Federated Unsupervised Representation Learning](https://arxiv.org/pdf/2010.08982) [J]. arXiv preprint arXiv:2010.08982.
- Yifan Luo, Jindan Xu, Wei Xu, Kezhi Wang .[Sliding Differential Evolution Scheduling for Federated Learning in Bandwidth-Limited Networks](https://arxiv.org/pdf/2010.08991) [J]. arXiv preprint arXiv:2010.08991.
- Sheng Shen, Tianqing Zhu, Di Wu, Wei Wang, Wanlei Zhou .[From Distributed Machine Learning To Federated Learning: In The View Of Data Privacy And Security](https://arxiv.org/pdf/2010.09258) [J]. arXiv preprint arXiv:2010.09258.
- Vatsal Patel, Sarth Kanani, Tapan Pathak, Pankesh Patel, Muhammad Intizar Ali, John Breslin .[A Demonstration of Smart Doorbell Design Using Federated Deep Learning](https://arxiv.org/pdf/2010.09687) [J]. arXiv preprint arXiv:2010.09687.
- Mohammad Mohammadi Amiri, Tolga M. Duman, Deniz Gunduz, Sanjeev R. Kulkarni, H. Vincent Poor .[Blind Federated Edge Learning](https://arxiv.org/pdf/2010.10030) [J]. arXiv preprint arXiv:2010.10030.
- Xinjian Luo, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi .[Feature Inference Attack on Model Predictions in Vertical Federated Learning](https://arxiv.org/pdf/2010.10152) [J]. arXiv preprint arXiv:2010.10152.
- [NIPS]Zhongxiang Dai, Kian Hsiang Low, Patrick Jaillet .[Federated Bayesian Optimization via Thompson Sampling](https://arxiv.org/pdf/2010.10154) [J]. arXiv preprint arXiv:2010.10154.
- Raed Abdel Sater, A. Ben Hamza .[A Federated Learning Approach to Anomaly Detection in Smart Buildings](https://arxiv.org/pdf/2010.10293) [J]. arXiv preprint arXiv:2010.10293.
- Yupeng Jiang, Yong Li, Yipeng Zhou, Xi Zheng .[Mitigating Sybil Attacks on Differential Privacy based Federated Learning](https://arxiv.org/pdf/2010.10572) [J]. arXiv preprint arXiv:2010.10572.
- Yifan Hu, Wei Xia, Jun Xiao, Chao Wu .[GFL: A Decentralized Federated Learning Framework Based On Blockchain](https://arxiv.org/pdf/2010.10996) [J]. arXiv preprint arXiv:2010.10996.
- Abhimanyu Dubey, Alex Pentland .[Differentially-Private Federated Linear Bandits](https://arxiv.org/pdf/2010.11425) [J]. arXiv preprint arXiv:2010.11425.
- Jinliang Yuan, Mengwei Xu, Xiao Ma, Ao Zhou, Xuanzhe Liu, Shangguang Wang .[Hierarchical Federated Learning through LAN-WAN Orchestration](https://arxiv.org/pdf/2010.11612) [J]. arXiv preprint arXiv:2010.11612.
- [NIPS]Othmane Marfoq, Chuan Xu, Giovanni Neglia, Richard Vidal .[Throughput-Optimal Topology Design for Cross-Silo Federated Learning](https://arxiv.org/pdf/2010.12229) [J]. arXiv preprint arXiv:2010.12229.<br>[code:[omarfoq/communication-in-cross-silo-fl](https://github.com/omarfoq/communication-in-cross-silo-fl)]
- Komal Krishna Mogilipalepu, Sumanth Kumar Modukuri, Amarlingam Madapu, Sundeep Prabhakar Chepuri .[Federated Deep Unfolding for Sparse Recovery](https://arxiv.org/pdf/2010.12616) [J]. arXiv preprint arXiv:2010.12616.
- Zhaowei Zhu, Jingxuan Zhu, Ji Liu, Yang Liu .[Federated Bandit: A Gossiping Approach](https://arxiv.org/pdf/2010.12763) [J]. arXiv preprint arXiv:2010.12763.
- Mingyang Chen, Wen Zhang, Zonggang Yuan, Yantao Jia, Huajun Chen .[FedE: Embedding Knowledge Graphs in Federated Setting](https://arxiv.org/pdf/2010.12882) [J]. arXiv preprint arXiv:2010.12882.
- Jiayi Wang, Shiqiang Wang, Rong-Rong Chen, Mingyue Ji .[Local Averaging Helps: Hierarchical Federated Learning and Convergence Analysis](https://arxiv.org/pdf/2010.12998) [J]. arXiv preprint arXiv:2010.12998.
- Wen Sun, Shiyu Lei, Lu Wang, Zhiqiang Liu, Yan Zhang .[Adaptive Federated Learning and Digital Twin for Industrial Internet of Things](https://arxiv.org/pdf/2010.13058) [J]. arXiv preprint arXiv:2010.13058.
- Wanli Ni, Yuanwei Liu, Zhaohui Yang, Hui Tian, Xuemin Shen .[Federated Learning in Multi-RIS Aided Systems](https://arxiv.org/pdf/2010.13333) [J]. arXiv preprint arXiv:2010.13333.
- Dick Carrillo, Lam Duc Nguyen, Pedro H. J. Nardelli, Evangelos Pournaras, Plinio Morita, Demóstenes Z. Rodríguez, Merim Dzaferagic, Harun Siljak, Alexander Jung, Laurent Hébert-Dufresne, Irene Macaluso, Mehar Ullah, Gustavo Fraidenraich, Petar Popovski .[Containing Future Epidemics with Trustworthy Federated Systems for Ubiquitous Warning and Response](https://arxiv.org/pdf/2010.13392) [J]. arXiv preprint arXiv:2010.13392.
- Elsa Rizk, Stefan Vlaski, Ali H. Sayed .[Optimal Importance Sampling for Federated Learning](https://arxiv.org/pdf/2010.13600) [J]. arXiv preprint arXiv:2010.13600.
- Wenlin Chen, Samuel Horvath, Peter Richtarik .[Optimal Client Sampling for Federated Learning](https://arxiv.org/pdf/2010.13723) [J]. arXiv preprint arXiv:2010.13723.
- Y. Sarcheshmehpour, M. Leinonen, A. Jung .[Federated Learning From Big Data Over Networks](https://arxiv.org/pdf/2010.14159) [J]. arXiv preprint arXiv:2010.14159.
- Jun Li, Lei Chen, Jiajia Chen .[Scalable Federated Learning over Passive Optical Networks](https://arxiv.org/pdf/2010.15454) [J]. arXiv preprint arXiv:2010.15454.
- Sudipan Saha, Tahir Ahmad .[Federated Transfer Learning: concept and applications](https://arxiv.org/pdf/2010.15561) [J]. arXiv preprint arXiv:2010.15561.
- Mustafa Safa Ozdayi, Murat Kantarcioglu, Rishabh Iyer .[Improving Accuracy of Federated Learning in Non-IID Settings](https://arxiv.org/pdf/2010.15582) [J]. arXiv preprint arXiv:2010.15582.
- Dhruv Guliani, Francoise Beaufays, Giovanni Motta .[Training Speech Recognition Models with Federated Learning: A Quality/Cost Framework](https://arxiv.org/pdf/2010.15965) [J]. arXiv preprint arXiv:2010.15965.
- Olga Zolotareva (1), Reza Nasirigerdeh (1), Julian Matschinske (1), Reihaneh Torkzadehmahani (1), Tobias Frisch (2), Julian Späth (1), David B. Blumenthal (1), Amir Abbasinejad (1 and 4), Paolo Tieri (3 nad 4), Nina K. Wenke (1), Markus List (1), Jan Baumbach (1 and 2) ((1) Chair of Experimental Bioinformatics, TUM School of Life Sciences, Technical University of Munich, Munich, Germany, (2) Department of Mathematics and Computer Science, University of Southern Denmark, Odense, Denmark, (3) CNR National Research Council, IAC Institute for Applied Computing, Rome, Italy, (4) SapienzaUniversity of Rome, Rome, Italy) .[Flimma: a federated and privacy-preserving tool for differential gene expression analysis](https://arxiv.org/pdf/2010.16403) [J]. arXiv preprint arXiv:2010.16403.
- Yuchen Zhao, Hanyang Liu, Honglin Li, Payam Barnaghi, Hamed Haddadi .[Semi-supervised Federated Learning for Activity Recognition](https://arxiv.org/pdf/2011.00851) [J]. arXiv preprint arXiv:2011.00851.
- Chen Wu, Xian Yang, Sencun Zhu, Prasenjit Mitra .[Mitigating Backdoor Attacks in Federated Learning](https://arxiv.org/pdf/2011.01767) [J]. arXiv preprint arXiv:2011.01767.
- Tiansheng Huang, Weiwei Lin, Wentai Wu, Ligang He, Keqin Li, Albert Y.Zomaya .[An Efficiency-boosting Client Selection Scheme for Federated Learning with Fairness Guarantee](https://arxiv.org/pdf/2011.01783) [J]. arXiv preprint arXiv:2011.01783.
- Kenneth Stewart, Yanqi Gu .[One-Shot Federated Learning with Neuromorphic Processors](https://arxiv.org/pdf/2011.01813) [J]. arXiv preprint arXiv:2011.01813.
- Zhaolin Ren, Aoxiao Zhong, Zhengyuan Zhou, Na Li .[Federated LQR: Learning through Sharing](https://arxiv.org/pdf/2011.01815) [J]. arXiv preprint arXiv:2011.01815.
- Sebastien Andreina, Giorgia Azzurra Marson, Helen Möllering, Ghassan Karame .[BaFFLe: Backdoor detection via Feedback-based Federated Learning](https://arxiv.org/pdf/2011.02167) [J]. arXiv preprint arXiv:2011.02167.
- Hyowoon Seo, Jihong Park, Seungeun Oh, Mehdi Bennis, Seong-Lyun Kim .[Federated Knowledge Distillation](https://arxiv.org/pdf/2011.02367) [J]. arXiv preprint arXiv:2011.02367.
- Zhihua Tian, Rui Zhang, Xiaoyang Hou, Jian Liu, Kui Ren .[FederBoost: Private Federated Learning for GBDT](https://arxiv.org/pdf/2011.02796) [J]. arXiv preprint arXiv:2011.02796.
- Junjie Pang, Jianbo Li, Zhenzhen Xie, Yan Huang, Zhipeng Cai .[Collaborative City Digital Twin For Covid-19 Pandemic: A Federated Learning Solution](https://arxiv.org/pdf/2011.02883) [J]. arXiv preprint arXiv:2011.02883.
- Ali Abedi, Shehroz S. Khan .[FedSL: Federated Split Learning on Distributed Sequential Data in Recurrent Neural Networks](https://arxiv.org/pdf/2011.03180) [J]. arXiv preprint arXiv:2011.03180.
- Gautham Krishna Gudur, Bala Shyamala Balaji, Satheesh K. Perepu .[Resource-Constrained Federated Learning with Heterogeneous Labels and Models](https://arxiv.org/pdf/2011.03206) [J]. arXiv preprint arXiv:2011.03206.
- Leye Wang, Han Yu, Xiao Han .[Federated Crowdsensing: Framework and Challenges](https://arxiv.org/pdf/2011.03208) [J]. arXiv preprint arXiv:2011.03208.
- Longfei Zheng, Jun Zhou, Chaochao Chen, Bingzhe Wu, Li Wang, Benyu Zhang .[ASFGNN: Automated Separated-Federated Graph Neural Network](https://arxiv.org/pdf/2011.03248) [J]. arXiv preprint arXiv:2011.03248.
- Nader Bouacida, Jiahui Hou, Hui Zang, Xin Liu .[Adaptive Federated Dropout: Improving Communication Efficiency and Generalization for Federated Learning](https://arxiv.org/pdf/2011.04050) [J]. arXiv preprint arXiv:2011.04050.
- Javad Ghareh Chamani (1), Dimitrios Papadopoulos (1) ((1) Hong Kong University of Science and Technology) .[Mitigating Leakage in Federated Learning with Trusted Hardware](https://arxiv.org/pdf/2011.04948) [J]. arXiv preprint arXiv:2011.04948.
- Zhibin Wang, Jiahang Qiu, Yong Zhou, Yuanming Shi, Liqun Fu, Wei Chen, Khaled B. Lataief .[Federated Learning via Intelligent Reflecting Surface](https://arxiv.org/pdf/2011.05051) [J]. arXiv preprint arXiv:2011.05051.
- Nguyen Truong, Kai Sun, Siyao Wang, Florian Guitton, Yike Guo .[Privacy Preservation in Federated Learning: Insights from the GDPR Perspective](https://arxiv.org/pdf/2011.05411) [J]. arXiv preprint arXiv:2011.05411.
- Raouf Kerkouche, Gergely Ács, Claude Castelluccia, Pierre Genevès .[Compression Boosts Differentially Private Federated Learning](https://arxiv.org/pdf/2011.05578) [J]. arXiv preprint arXiv:2011.05578.
- Xiaowen Cao, Guangxu Zhu, Jie Xu, Shuguang Cui .[Optimized Power Control for Over-the-Air Federated Edge Learning](https://arxiv.org/pdf/2011.05587) [J]. arXiv preprint arXiv:2011.05587.
- Jiangcheng Qin, Baisong Liu .[A Novel Privacy-Preserved Recommender System Framework based on Federated Learning](https://arxiv.org/pdf/2011.05614) [J]. arXiv preprint arXiv:2011.05614.
- Saurav Prakash, Sagar Dhakal, Mustafa Akdeniz, Yair Yona, Shilpa Talwar, Salman Avestimehr, Nageen Himayat .[Coded Computing for Low-Latency Federated Learning over Wireless Edge Networks](https://arxiv.org/pdf/2011.06223) [J]. arXiv preprint arXiv:2011.06223.
- Dipankar Sarkar, Ankur Narang, Sumit Rai .[Fed-Focal Loss for imbalanced data classification in Federated Learning](https://arxiv.org/pdf/2011.06283) [J]. arXiv preprint arXiv:2011.06283.
- Lixuan Yang, Cedric Beliard, Dario Rossi .[Heterogeneous Data-Aware Federated Learning](https://arxiv.org/pdf/2011.06393) [J]. arXiv preprint arXiv:2011.06393.
- Shuhao Xia, Jingyang Zhu, Yuhan Yang, Yong Zhou, Yuanming Shi, Wei Chen .[Fast Convergence Algorithm for Analog Federated Learning](https://arxiv.org/pdf/2011.06658) [J]. arXiv preprint arXiv:2011.06658.
- Anna Bogdanova, Akie Nakai, Yukihiko Okada, Akira Imakura, Tetsuya Sakurai .[Federated Learning System without Model Sharing through Integration of Dimensional Reduced Data Representations](https://arxiv.org/pdf/2011.06803) [J]. arXiv preprint arXiv:2011.06803.
- Jiyue Huang, Rania Talbi, Zilong Zhao, Sara Boucchenak, Lydia Y. Chen, Stefanie Roos .[An Exploratory Analysis on Users' Contributions in Federated Learning](https://arxiv.org/pdf/2011.06830) [J]. arXiv preprint arXiv:2011.06830.
- Ahmet M. Elbir .[Hybrid Federated and Centralized Learning](https://arxiv.org/pdf/2011.06892) [J]. arXiv preprint arXiv:2011.06892.
- Mohammad Bakhtiari, Reza Nasirigerdeh, Reihaneh Torkzadehmahani, Amirhossein Bayat, David B. Blumenthal, Markus List, Jan Baumbach .[Federated Multi-Mini-Batch: An Efficient Training Approach to Federated Learning in Non-IID Environments](https://arxiv.org/pdf/2011.07006) [J]. arXiv preprint arXiv:2011.07006.
- Huiwen Wu, Cen Chen, Li Wang .[A Theoretical Perspective on Differentially Private Federated Multi-task Learning](https://arxiv.org/pdf/2011.07179) [J]. arXiv preprint arXiv:2011.07179.
- Dipankar Sarkar, Sumit Rai, Ankur Narang .[CatFedAvg: Optimising Communication-efficiency and Classification Accuracy in Federated Learning](https://arxiv.org/pdf/2011.07229) [J]. arXiv preprint arXiv:2011.07229.
- Mahdi Boloursaz Mashhadi, Nir Shlezinger, Yonina C. Eldar, Deniz Gunduz .[FedRec: Federated Learning of Universal Receivers over Fading Channels](https://arxiv.org/pdf/2011.07271) [J]. arXiv preprint arXiv:2011.07271.
- Anbu Huang .[Dynamic backdoor attacks against federated learning](https://arxiv.org/pdf/2011.07429) [J]. arXiv preprint arXiv:2011.07429.
- Harry Cai, Daniel Rueckert, Jonathan Passerat-Palmbach .[2CP: Decentralized Protocols to Transparently Evaluate Contributivity in Blockchain Federated Learning Environments](https://arxiv.org/pdf/2011.07516) [J]. arXiv preprint arXiv:2011.07516.
- Honglin Yuan, Manzil Zaheer, Sashank Reddi .[Federated Composite Optimization](https://arxiv.org/pdf/2011.08474) [J]. arXiv preprint arXiv:2011.08474.
- Burak Hasircioglu, Deniz Gunduz .[Private Wireless Federated Learning with Anonymous Over-the-Air Computation](https://arxiv.org/pdf/2011.08579) [J]. arXiv preprint arXiv:2011.08579.
- Tiansheng Huang, Weiwei Lin, Keqin Li, Albert Y. Zomaya .[Stochastic Client Selection for Federated Learning with Volatile Clients](https://arxiv.org/pdf/2011.08756) [J]. arXiv preprint arXiv:2011.08756.
- Haiqin Weng, Juntao Zhang, Feng Xue, Tao Wei, Shouling Ji, Zhiyuan Zong .[Privacy Leakage of Real-World Vertical Federated Learning](https://arxiv.org/pdf/2011.09290) [J]. arXiv preprint arXiv:2011.09290.
- Nick Angelou, Ayoub Benaissa, Bogdan Cebere, William Clark, Adam James Hall, Michael A. Hoeh, Daniel Liu, Pavlos Papadopoulos, Robin Roehm, Robert Sandmann, Phillipp Schoppmann, Tom Titcombe .[Asymmetric Private Set Intersection with Applications to Contact Tracing and Private Vertical Federated Machine Learning](https://arxiv.org/pdf/2011.09350) [J]. arXiv preprint arXiv:2011.09350.
- Nicolas Kourtellis, Kleomenis Katevas, Diego Perino .[FLaaS: Federated Learning as a Service](https://arxiv.org/pdf/2011.09359) [J]. arXiv preprint arXiv:2011.09359.
- Di Chai, Leye Wang, Kai Chen, Qiang Yang .[FedEval: A Benchmark System with a Comprehensive Evaluation Model for Federated Learning](https://arxiv.org/pdf/2011.09655) [J]. arXiv preprint arXiv:2011.09655.
- Ihab Mohammed, Shadha Tabatabai, Ala Al-Fuqaha, Faissal El Bouanani, Junaid Qadir, Basheer Qolomany, Mohsen Guizani .[Budgeted Online Selection of Candidate IoT Clients to Participate in Federated Learning](https://arxiv.org/pdf/2011.09849) [J]. arXiv preprint arXiv:2011.09849.
- Yunlong Lu, Xiaohong Huang, Ke Zhang, Sabita Maharjan, Yan Zhang .[Low-latency Federated Learning and Blockchain for Edge Association in Digital Twin empowered 6G Networks](https://arxiv.org/pdf/2011.09902) [J]. arXiv preprint arXiv:2011.09902.
- Hang Liu, Xiaojun Yuan, Ying-Jun Angela Zhang .[Reconfigurable Intelligent Surface Enabled Federated Learning: A Unified Communication-Learning Design Approach](https://arxiv.org/pdf/2011.10282) [J]. arXiv preprint arXiv:2011.10282.
- Xinyi Xu, Lingjuan Lyu .[Towards Building a Robust and Fair Federated Learning System](https://arxiv.org/pdf/2011.10464) [J]. arXiv preprint arXiv:2011.10464.
- Hong Lin, Lidan Shou, Ke Chen, Gang Chen, Sai Wu .[LINDT: Tackling Negative Federated Learning with Local Adaptation](https://arxiv.org/pdf/2011.11160) [J]. arXiv preprint arXiv:2011.11160.
- Miao Yang, Akitanoshou Wong, Hongbin Zhu, Haifeng Wang, Hua Qian .[Federated learning with class imbalance reduction](https://arxiv.org/pdf/2011.11266) [J]. arXiv preprint arXiv:2011.11266.
- Yilun Lin, Chaochao Chen, Cen Chen, Li Wang .[Improving Federated Relational Data Modeling via Basis Alignment and Weight Penalty](https://arxiv.org/pdf/2011.11369) [J]. arXiv preprint arXiv:2011.11369.
- Dong Yang, Ziyue Xu, Wenqi Li, Andriy Myronenko, Holger R. Roth, Stephanie Harmon, Sheng Xu, Baris Turkbey, Evrim Turkbey, Xiaosong Wang, Wentao Zhu, Gianpaolo Carrafiello, Francesca Patella, Maurizio Cariati, Hirofumi Obinata, Hitoshi Mori, Kaku Tamura, Peng An, Bradford J. Wood, Daguang Xu .[Federated Semi-Supervised Learning for COVID Region Segmentation in Chest CT using Multi-National Data from China, Italy, Japan](https://arxiv.org/pdf/2011.11750) [J]. arXiv preprint arXiv:2011.11750.
- Minh N. H. Nguyen, Nguyen H. Tran, Yan Kyaw Tun, Zhu Han, Choong Seon Hong .[Toward Multiple Federated Learning Services Resource Sharing in Mobile Edge Networks](https://arxiv.org/pdf/2011.12469) [J]. arXiv preprint arXiv:2011.12469.
- Sen Lin, Li Yang, Zhezhi He, Deliang Fan, Junshan Zhang .[MetaGater: Fast Learning of Conditional Channel Gated Networks via Federated Meta-Learning](https://arxiv.org/pdf/2011.12511) [J]. arXiv preprint arXiv:2011.12511.
- Hangyu Zhu, Rui Wang, Yaochu Jin, Kaitai Liang, Jianting Ning .[Distributed Additive Encryption and Quantization for Privacy Preserving Federated Deep Learning](https://arxiv.org/pdf/2011.12623) [J]. arXiv preprint arXiv:2011.12623.
- Yong Xiao, Yingyu Li, Guangming Shi, H. Vincent Poor .[Optimizing Resource-Efficiency for Federated Edge Intelligence in IoT Networks](https://arxiv.org/pdf/2011.12691) [J]. arXiv preprint arXiv:2011.12691.
- Helin Yang, Jun Zhao, Zehui Xiong, Kwok-Yan Lam, Sumei Sun, Liang Xiao .[Privacy-Preserving Federated Learning for UAV-Enabled Networks: Learning-Based Joint Scheduling and Resource Management](https://arxiv.org/pdf/2011.14197) [J]. arXiv preprint arXiv:2011.14197.
- Chandra Thapa, M.A.P. Chamikara, Seyit A. Camtepe .[Advancements of federated learning towards privacy preservation: from federated learning to split learning](https://arxiv.org/pdf/2011.14818) [J]. arXiv preprint arXiv:2011.14818.


## Blogs && Tutorials
- [Learn to adapt Flower for your use-case](https://flower.dev/blog)
- [Flower](https://flower.dev/docs/example_walkthrough_pytorch_mnist.html)
- [Online Comic from Google AI on Federated Learning](https://federated.withgoogle.com/)
- [PPT][thormacy/Federated-Learning](https://github.com/thormacy/Federated-Learning/tree/master/PPT)
- [Federated Learning: Collaborative Machine Learning without Centralized Training Data](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html) - Google AI Blog 2017
- [Under The Hood of The Pixel 2: How AI Is Supercharging Hardware](https://ai.google/stories/ai-in-hardware/)
- [An Introduction to Federated Learning](http://vision.cloudera.com/an-introduction-to-federated-learning/)
- [Federated learning: Distributed machine learning with data locality and privacy](https://blog.fastforwardlabs.com/2018/11/14/federated-learning.html)
- [Federated Learning: The Future of Distributed Machine Learning](https://medium.com/syncedreview/federated-learning-the-future-of-distributed-machine-learning-eec95242d897)
- [Federated Learning for Wake Word Detection](https://medium.com/snips-ai/federated-learning-for-wake-word-detection-c8b8c5cdd2c5)
- [An Open Framework for Secure and Privated AI](https://medium.com/@ODSC/an-open-framework-for-secure-and-private-ai-96c1891a4b)
- [A Brief Introduction to Differential Privacy](https://medium.com/georgian-impact-blog/a-brief-introduction-to-differential-privacy-eacf8722283b)
- [An Overview of Federated Learning](https://medium.com/datadriveninvestor/an-overview-of-federated-learning-8a1a62b0600d). This blog introduces some challenges of federated learning, including *Inference Attack* and *Model Poisoning*.
- [PySyft](https://github.com/OpenMined/PySyft/tree/dev/examples/tutorials)
- [tensorflow TFF](https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification)
- [open-intelligence/federated-learning-chinese](https://github.com/open-intelligence/federated-learning-chinese)
- [杨强：联邦学习](https://mp.weixin.qq.com/s/5FTrG5SZey2yeIbuyT3HoQ)
- [联邦学习的研究及应用](https://mp.weixin.qq.com/s?src=11&timestamp=1555896266&ver=1561&signature=ZtLlc7qakNAdw8hV3dxaB30PxtK9hAshYsIxccFf-D4eJrUw6YKQcqD0lD3SDMEn4egQTafUZr429er7SueP6HKLTr*uFKfr6JuHc3OvfdJ-uExiEJStHFynC65htbLp&new=1)
- [杨强：GDPR对AI的挑战和基于联邦迁移学习的对策](https://zhuanlan.zhihu.com/p/42646278)
- [联邦学习的研究与应用](https://aisp-1251170195.file.myqcloud.com/fedweb/1553845987342.pdf)
- [Federated Learning and Transfer Learning for Privacy, Security and Confidentiality](https://aisp-1251170195.file.myqcloud.com/fedweb/1552916850679.pdf) (AAAI-19)
- [GDPR, Data Shortage and AI](https://aisp-1251170195.file.myqcloud.com/fedweb/1552916659436.pdf) (AAAI-19)
- [GDPR, Data Shortage and AI](https://aaai.org/Conferences/AAAI-19/invited-speakers/#yang) (AAAI-19 Invited Talk)
- [video][GDPR, Data Shortage and AI](https://vimeo.com/313941621) - Qiang Yang, AAAI 2019 Invited Talk
- [谷歌发布全球首个产品级移动端分布式机器学习系统，数千万手机同步训练](https://www.jiemian.com/article/2853096.html)
- [clara-federated-learning](https://blogs.nvidia.com/blog/2019/12/01/clara-federated-learning/)
- [What is Federated Learning](https://blogs.nvidia.com/blog/2019/10/13/what-is-federated-learning/) - Nvidia 2019
- [nvidia-uses-federated-learning-to-create-medical-imaging-ai](https://venturebeat.com/2019/10/13/nvidia-uses-federated-learning-to-create-medical-imaging-ai/)
- [federated-learning-technique-predicts-hospital-stay-and-patient-mortality](https://venturebeat.com/2019/03/25/federated-learning-technique-predicts-hospital-stay-and-patient-mortality/)
- [pubmed](https://www.ncbi.nlm.nih.gov/pubmed/29500022)
- [google-mayo-clinic-partnership-patient-data](https://www.statnews.com/2019/09/10/google-mayo-clinic-partnership-patient-data/)
- [webank-clustar](https://www.digfingroup.com/webank-clustar/)
- [Private AI-Federated Learning with PySyft and PyTorch](https://towardsdatascience.com/private-ai-federated-learning-with-pysyft-and-pytorch-954a9e4a4d4e)
- [Federated Learning in 10 lines of PyTorch and PySyft](https://blog.openmined.org/upgrade-to-federated-learning-in-10-lines/)
- [A beginners Guided to Federated Learning](https://hackernoon.com/a-beginners-guide-to-federated-learning-b29e29ba65cf). Federated Learning was born at the intersection of on-device AI, blockchain, and edge computing/IoT.
- [video][Federated Learning: Machine Learning on Decentralized Data (Google I/O'19)](https://www.youtube.com/watch?v=89BGjQYA0uE)
- [video][TensorFlow Federated (TFF): Machine Learning on Decentralized Data ](https://www.youtube.com/watch?v=1YbPmkChcbo)
- [video][Federated Learning: Machine Learning on Decentralized Data](https://www.youtube.com/watch?v=89BGjQYA0uE)
- [video][Federated Learning](https://www.youtube.com/watch?v=xJkY3ehX_MI)
- [video][Making every phone smarter with Federated Learning](https://www.youtube.com/watch?v=gbRJPa9d-VU) - Google, 2018
- [video][Secure and Private AI Udacity](https://classroom.udacity.com/courses/ud185)


## Framework
- [FederatedAI/FATE](https://github.com/FederatedAI/FATE)
- [jd-9n/9nfl](https://github.com/jd-9n/9nfl)
- [tensorflow/federated](https://github.com/tensorflow/federated)
- [bytedance/fedlearner](https://github.com/bytedance/fedlearner)
- [FedML-AI/FedML](https://github.com/FedML-AI/FedML)
- [IBM/federated-learning-lib](https://github.com/IBM/federated-learning-lib)
- [OpenMined/PySyft](https://github.com/OpenMined/PySyft)
- [PaddlePaddle/PaddleFL](https://github.com/PaddlePaddle/PaddleFL)
- [flower](https://flower.dev/)
- [facebookresearch/CrypTen](https://github.com/facebookresearch/CrypTen)


## Projects
- [shashigharti/federated-learning-on-raspberry-pi](https://github.com/shashigharti/federated-learning-on-raspberry-pi)
- [shaoxiongji/federated-learning](https://github.com/shaoxiongji/federated-learning)
- [mccorby](https://github.com/mccorby)
- [roxanneluo/Federated-Learning](https://github.com/roxanneluo/Federated-Learning)
- [dvc](https://dvc.org/) # unknown
- [papersdclub/Differentially_private_federated_learning](https://github.com/papersdclub/Differentially_private_federated_learning)
- [AshwinRJ/Federated-Learning-PyTorch](https://github.com/AshwinRJ/Federated-Learning-PyTorch)


## Datasets && Benchmark
- [Federated iNaturalist/Landmarks](https://github.com/google-research/google-research/tree/master/federated_vision_datasets)
- [DIDL][A Performance Evaluation of Federated Learning Algorithms](https://www.researchgate.net/profile/Gregor_Ulm/publication/329106719_A_Performance_Evaluation_of_Federated_Learning_Algorithms/links/5c0fabcfa6fdcc494febf907/A-Performance-Evaluation-of-Federated-Learning-Algorithms.pdf)
- Gregor Ulm, Emil Gustavsson, Mats Jirstrand .[Functional Federated Learning in Erlang (ffl-erl)](https://arxiv.org/pdf/1808.08143) [J]. arXiv preprint arXiv:1808.08143.
- Caldas S, Duddu S M K, Wu P, et al. [Leaf: A benchmark for federated settings](https://arxiv.org/abs/1812.01097)[J]. arXiv preprint arXiv:1812.01097, 2018. <br> [code:[Github](https://github.com/TalwalkarLab/leaf);[website](https://leaf.cmu.edu/)]
- [Edge AIBench: Towards Comprehensive End-to-end Edge Computing Benchmarking](https://arxiv.org/abs/1908.01924)
- Jiahuan Luo, Xueyang Wu, Yun Luo, Anbu Huang, Yunfeng Huang, Yang Liu, Qiang Yang .[Real-World Image Datasets for Federated Learning](https://arxiv.org/pdf/1910.11089) [J]. arXiv preprint arXiv:1910.11089.
- Yang Liu, Zhuo Ma, Ximeng Liu, Zhuzhu Wang, Siqi Ma, Ken Ren .[Revocable Federated Learning: A Benchmark of Federated Forest](https://arxiv.org/pdf/1911.03242) [J]. arXiv preprint arXiv:1911.03242.
- Vaikkunth Mugunthan, Anton Peraire-Bueno, Lalana Kagal .[PrivacyFL: A simulator for privacy-preserving and secure federated learning](https://arxiv.org/pdf/2002.08423) [J]. arXiv preprint arXiv:2002.08423.
- Lifeng Liu, Fengda Zhang, Jun Xiao, Chao Wu .[Evaluation Framework For Large-scale Federated Learning](https://arxiv.org/pdf/2003.01575) [J]. arXiv preprint arXiv:2003.01575.
- Sixu Hu, Yuan Li, Xu Liu, Qinbin Li, Zhaomin Wu, Bingsheng He .[The OARF Benchmark Suite: Characterization and Implications for Federated Learning Systems](https://arxiv.org/pdf/2006.07856) [J]. arXiv preprint arXiv:2006.07856.
- Weiming Zhuang, Yonggang Wen, Xuesen Zhang, Xin Gan, Daiying Yin, Dongzhan Zhou, Shuai Zhang, Shuai Yi .[Performance Optimization for Federated Person Re-identification via Benchmark Analysis](https://arxiv.org/pdf/2008.11560) [J]. arXiv preprint arXiv:2008.11560. <br> [code:[cap-ntu/FedReID](https://github.com/cap-ntu/FedReID)]



## Scholars
- [Yang Qiang](https://scholar.google.com/citations?hl=en&user=1LxWZLQAAAAJ)
- [H. Brendan McMahan](https://scholar.google.com/citations?user=iKPWydkAAAAJ&hl=en)
- [jakub konečný](https://scholar.google.com/citations?user=4vq7eXQAAAAJ&hl=en)
- [H. Vincent Poor](https://ee.princeton.edu/people/h-vincent-poor)
- [Hao Ye](https://scholar.google.ca/citations?user=ok7OWEAAAAAJ&hl=en)
- [Ye Li](http://liye.ece.gatech.edu/)



## Conferences and Workshops
- [FL-ICML 2020](http://federated-learning.org/fl-icml-2020/) - Organized by IBM Watson Research.
- [FL-IBM 2020](https://federated-learning.bitbucket.io/ibm2020/) - Organized by IBM Watson Research and Webank.
- [FL-NeurIPS 2019](http://federated-learning.org/fl-neurips-2019/) - Organized by Google, Webank, NTU, CMU.
- [FL-IJCAI 2019](https://www.ijcai19.org/workshops.html) - Organized by Webank.
- [Google Federated Learning workshop](https://sites.google.com/view/federated-learning-2019/home) - Organized by Google.


## Company
- [Adap](https://adap.com/en)
- [Snips](https://snips.ai/); [Snips](https://www.theverge.com/2019/11/21/20975607/sonos-buys-snips-ai-voice-assistant-privacy)
- [Privacy.ai](https://privacy.ai/)
- [OpenMined](https://www.openmined.org/)
- [Arkhn](https://arkhn.org/en/)
- [Scaleout](https://scaleoutsystems.com/)
- [MELLODDY](https://www.melloddy.eu/)
- [DataFleets](https://www.datafleets.com/)
- [baidu PaddleFL](https://github.com/PaddlePaddle/PaddleFL)
- [Owkin](https://owkin.com/): Medical research
- [XAIN](https://www.xain.io/) [[Github]](https://github.com/xainag/xain-fl): Automated Invoicing
- [S20](https://www.s20.ai/): Multiple third party collaboration
- [google TensorFlow](https://github.com/tensorflow/federated)
- [bytedance](https://github.com/bytedance/fedlearner)
- [JD](https://github.com/jd-9n/9nfl)
- [平安蜂巢](.)
- [nvidia clare](https://developer.nvidia.com/clara) 
- [huawei NAIE](https://console.huaweicloud.com/naie/)
- [冰鉴](.)
- [数犊科技](https://www.sudoprivacy.com/)
- [同态科技-迷雾计算](https://www.ttaicloud.com/)
- [TalkingData](https://sdmk.talkingdata.com/#/home/datasecurity)
- [融数联智](https://www.udai.link/)
- [算数力科技-CompuTa](https://www.computa.com/)
- [摩联科技](https://www.aitos.io/index/index/index.html)
- [ARPA-ARPA隐私计算协议](https://arpachain.io/)
- [趣链科技-BitXMesh可信数据网络](https://bitxmesh.com/)

### 联邦学习工具基础能力测评
| 公司 | 产品 |认证|
| :-----:| :----: |:---:|
|同盾控股有限公司 |[同盾智邦知识联邦平台](https://www.tongdun.cn/ai/solution/aiknowledge)|信通院认证|
|腾讯云计算(北京)有限责任公司| 腾讯神盾Angel PowerFL联邦计算平台|信通院认证|
|翼健（上海）信息科技有限公司| [翼数坊XDP隐私安全计算平台](https://www.basebit.me/)|信通院认证|
|京东云计算有限公司| 京东智联云联邦学习平台|信通院认证|
|京东数科海益信息科技有限公司| [联邦模盒](https://www.jddglobal.com/products/union-learn)|信通院认证|
|杭州锘崴信息科技有限公司| [锘崴信联邦学习平台](https://www.nvxclouds.com/)|信通院认证|
|[深圳前海新心数字科技有限公司](https://www.xinxindigits.com/about/services)| 新心数述联邦学习平台|信通院认证|
|深圳前海微众银行股份有限公司| [联邦学习云服务平台](https://cn.fedai.org/)|信通院认证|
|上海富数科技有限公司| [阿凡达安全计算平台](https://www.fudata.cn/federated-machine-learning)|信通院认证|
|天翼电子商务有限公司| CTFL天翼联邦学习平台|信通院认证|
|中国电信股份有限公司云计算分公司| 天翼云诸葛AI-联邦学习平台|信通院认证|
|厦门渊亭信息科技有限公司| [DataExa-Insight人工智能中台系统](http://www.dataexa.com/product/insight)|信通院认证|
|光之树（北京）科技有限公司| [云间联邦学习平台](https://www.guangzhishu.com/)|信通院认证|
|神谱科技（上海）有限公司| [神谱科技Seceum联邦学习系统](http://www.seceum.com/home.html)|信通院认证|
|深圳市洞见智慧科技有限公司| [洞见数智联邦平台（INSIGHTONE）](https://www.insightone.cn/)|信通院认证|
|[星环信息科技（上海）有限公司](https://www.transwarp.io/transwarp/index.html)| 星环联邦学习软件|信通院认证|
|华控清交信息科技（北京）有限公司| [清交PrivPy多方计算平台](https://www.tsingj.com/)|信通院认证|
|腾讯云计算（北京）有限责任公司 |腾讯云联邦学习应用平台软件|信通院认证|


``` [腾讯fele](https://cloud.tencent.com/product/fele)```

### 多方安全计算工具基础能力评测
| 公司 | 产品 |认证|
| :-----:| :----: |:---:|
|蓝象智联（杭州）科技有限公司| GAIA·Edge (保护数据隐私的多方安全计算产品)|信通院认证|
|腾讯云计算（北京）有限责任公司| 腾讯神盾Angel PowerFL联邦计算平台|信通院认证|
|深圳前海微众银行股份有限公司| [联邦学习云服务平台](https://cn.fedai.org/)|信通院认证|
|上海富数科技有限公司| [阿凡达安全计算平台](https://www.fudata.cn/federated-machine-learning)|信通院认证|
|矩阵元技术（深圳）有限公司| [矩阵元隐私计算服务系统](https://jugo.juzix.net/home)|信通院认证|
|蚂蚁区块链科技(上海)有限公司| [蚂蚁链摩斯安全计算平台（MORSE）](https://antchain.antgroup.com/products/morse)|信通院认证|

### 可信执行环境计算平台基础能力评测
| 公司 | 产品 |认证|
| :-----:| :----: |:---:|
|北京冲量在线科技有限公司| [冲量数据互联平台](http://www.impulse.top/)|信通院认证|
|翼健（上海）信息科技有限公司| [翼数坊XDP隐私安全计算平台](https://www.basebit.me/)|信通院认证|
|上海隔镜信息科技有限公司| [天禄多方安全计算平台](https://www.trustmirror.com/product/index.html)|信通院认证|
|杭州锘崴信息科技有限公司| [锘崴信联邦学习平台](https://www.nvxclouds.com/)|信通院认证|
|蚂蚁智信（杭州）信息技术有限公司| [共享智能平台](https://blockchain.antgroup.com/solutions/tdsp)|信通院认证|
|华为技术有限公司| [可信智能计算服务TICS](https://www.huaweicloud.com/product/tics.html)|信通院认证|
|蚂蚁区块链科技（上海）有限公司|  [蚂蚁链数据隐私服务](https://blockchain.antgroup.com/products/openchain)|信通院认证|
