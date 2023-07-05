# Tricks Gallery

## Usage

In this section, we will show you how to run these tricks in detail.

```
cd trick-gallery
```

**DS**={`cifar-10`, `cifar-100`}; **TYPE**={`symmetric`, `asymmetric`,` pairflip`, `aggre`, `worst`, `noisy100`}; **RATE** :Noise Rate 

- Baseline

  ```bash
  python main.py --dataset {DS} --noise_type {TYPE} --noise_rate {RATE}
  ```

- ResResNet

  ```bash
  python main.py --dataset {DS} --noise_type {TYPE} --noise_rate {RATE} --net 'PreResNet18'
  ```

- Mixup

  ```bash
  python main.py --dataset {DS} --noise_type {TYPE} --noise_rate {RATE} --alpha 1
  ```

- Label Smoothing

  ```bash
  python main.py --dataset {DS} --noise_type {TYPE} --noise_rate {RATE} --label-smoothing 0.1
  ```

- Autoaug

  ```bash
  python main.py --dataset {DS} --noise_type {TYPE} --noise_rate {RATE} --aug_type 'autoaug'
  ```

- Randaug

  ```bash
  python main.py --dataset {DS} --noise_type {TYPE} --noise_rate {RATE} --aug_type 'randaug'
  ```

- Classbalance

  ```bash
  python main.py --dataset {DS} --noise_type {TYPE} --noise_rate {RATE} --class_balance 1
  ```




## Experiments

Vallina is an ordinary model, and EMA is a student model obtained by exponential moving average. The experiment was repeated three times, and mean±std was taken as the final result. 

**CIFAR-10**

<table class="tg">
<thead>
  <tr>
    <th  align="center" class="tg-baqh" colspan="2">Dataset</th>
    <th  align="center" class="tg-baqh" colspan="8">CIFAR-10</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td  align="center" class="tg-baqh" colspan="2">Noise type</td>
    <td  align="center" class="tg-baqh" colspan="3">Symmetric</td>
    <td  align="center" class="tg-baqh">Asym.</td>
    <td  align="center" class="tg-baqh" colspan="2">Instance</td>
    <td  align="center" class="tg-baqh" colspan="2">Real</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh" colspan="2">Noise rate</td>
    <td  align="center" class="tg-baqh">0.2</td>
    <td  align="center" class="tg-baqh">0.4</td>
    <td  align="center" class="tg-baqh">0.6</td>
    <td  align="center" class="tg-baqh">0.4</td>
    <td  align="center" class="tg-baqh">0.2</td>
    <td  align="center" class="tg-baqh">0.4</td>
    <td  align="center" class="tg-baqh">aggre</td>
    <td  align="center" class="tg-baqh">worst</td>
  </tr>
  <!-- baseline -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="2">Baseline</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">86.82±0.12</td>
    <td  align="center" class="tg-baqh">82.58±0.50</td>
    <td  align="center" class="tg-baqh">75.13±0.21</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">85.49±0.75</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">86.14±0.21</td>
    <td  align="center" class="tg-baqh">76.25±0.64</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">90.40±0.13</td>
    <td  align="center" class="tg-baqh">79.53±0.26</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">91.06±0.03</td>
    <td  align="center" class="tg-baqh">87.92±0.20</td>
    <td  align="center" class="tg-baqh">81.17±0.35</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">88.85±0.08</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">90.80±0.11</td>
    <td  align="center" class="tg-baqh">82.55±0.07</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">91.46±0.03</td>
    <td  align="center" class="tg-baqh">83.64±0.10</td>
  </tr>
  <!-- PreResNet -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="2">PreResNet</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">86.82±0.04</td>
    <td  align="center" class="tg-baqh">82.25±0.41</td>
    <td  align="center" class="tg-baqh">74.34±0.10</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">84.98±1.16</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">86.67±0.48</td>
    <td  align="center" class="tg-baqh">77.28±0.83</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">89.95±0.13</td>
    <td  align="center" class="tg-baqh">78.76±0.74</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">90.92±0.11</td>
    <td  align="center" class="tg-baqh">87.67±0.27</td>
    <td  align="center" class="tg-baqh">80.39±0.26</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">89.42±0.33</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">90.72±0.10</td>
    <td  align="center" class="tg-baqh">80.94±0.43</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">91.33±0.09</td>
    <td  align="center" class="tg-baqh">83.60±0.17</td>
  </tr>
  <!-- mixup -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="2">Mixup</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">90.07±0.07</td>
    <td  align="center" class="tg-baqh">86.44±0.13</td>
    <td  align="center" class="tg-baqh">79.28±0.14</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">88.22±0.40</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">90.30±0.10</td>
    <td  align="center" class="tg-baqh">78.82±0.42</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">92.08±0.13</td>
    <td  align="center" class="tg-baqh">82.50±0.25</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">92.69±0.13</td>
    <td  align="center" class="tg-baqh">89.98±0.00</td>
    <td  align="center" class="tg-baqh">84.13±0.05</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">90.68±0.05</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">92.97±0.10</td>
    <td  align="center" class="tg-baqh">83.28±0.28</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">93.29±0.09</td>
    <td  align="center" class="tg-baqh">85.54±0.10</td>
  </tr>
   <!-- labelsmoothing -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="2">Label Smoothing</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">87.37±0.23</td>
    <td  align="center" class="tg-baqh">83.09±0.57</td>
    <td  align="center" class="tg-baqh">76.04±0.43</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">86.15±0.29</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">86.70±0.29</td>
    <td  align="center" class="tg-baqh">76.94±0.82</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">90.67±0.05</td>
    <td  align="center" class="tg-baqh">79.52±0.29</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">91.18±0.15</td>
    <td  align="center" class="tg-baqh">88.05±0.22</td>
    <td  align="center" class="tg-baqh">81.76±0.25</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">90.11±0.20</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">91.13±0.07</td>
    <td  align="center" class="tg-baqh">83.29±0.25</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">91.85±0.09</td>
    <td  align="center" class="tg-baqh">83.55±0.21</td>
  </tr>
  <!-- Autoaug -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="2">Autoaug</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">91.37±0.03</td>
    <td  align="center" class="tg-baqh">87.18±0.12</td>
    <td  align="center" class="tg-baqh">80.30±0.22</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">88.54±0.44</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">90.47±0.43</td>
    <td  align="center" class="tg-baqh">76.47±0.73</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">93.66±0.08</td>
    <td  align="center" class="tg-baqh">83.79±0.59</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">93.26±0.13</td>
    <td  align="center" class="tg-baqh">90.43±0.16</td>
    <td  align="center" class="tg-baqh">84.60±0.12</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">90.70±0.06</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">92.57±0.09</td>
    <td  align="center" class="tg-baqh">78.75±0.30</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">94.08±0.04</td>
    <td  align="center" class="tg-baqh">86.41±0.06</td>
  </tr>
  <!-- Randaug -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="2">Randaug</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">91.06±0.10</td>
    <td  align="center" class="tg-baqh">86.92±0.53</td>
    <td  align="center" class="tg-baqh">79.13±0.58</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">88.38±0.23</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">89.42±0.24</td>
    <td  align="center" class="tg-baqh">74.89±2.00</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">93.44±0.01</td>
    <td  align="center" class="tg-baqh">83.50±0.36</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">93.18±0.03</td>
    <td  align="center" class="tg-baqh">90.21±0.15</td>
    <td  align="center" class="tg-baqh">83.68±0.14</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">90.41±0.18</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">92.09±0.12</td>
    <td  align="center" class="tg-baqh">76.10±0.19</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">93.96±0.05</td>
    <td  align="center" class="tg-baqh">85.97±0.10</td>
  </tr>
  <!-- Class-balance -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="2">Class Balance</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">86.61±0.55</td>
    <td  align="center" class="tg-baqh">82.71±0.16</td>
    <td  align="center" class="tg-baqh">75.54±0.88</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">86.90±0.18</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">87.20±0.38</td>
    <td  align="center" class="tg-baqh">81.53±0.31</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">90.64±0.10</td>
    <td  align="center" class="tg-baqh">79.59±0.34</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">90.91±0.10</td>
    <td  align="center" class="tg-baqh">88.10±0.28</td>
    <td  align="center" class="tg-baqh">81.29±0.36</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">89.74±0.12</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">91.44±0.12</td>
    <td  align="center" class="tg-baqh">86.15±0.14</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">91.50±0.04</td>
    <td  align="center" class="tg-baqh">83.80±0.24</td>
  </tr>
</tbody>
</table>



**CIFAR-100**




<table class="tg">
<thead>
  <tr>
    <th  align="center" class="tg-baqh" colspan="2">Dataset</th>
    <th  align="center" class="tg-baqh" colspan="7">CIFAR-100</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td  align="center" class="tg-baqh" colspan="2">Noise type</td>
    <td  align="center" class="tg-baqh" colspan="3">Symmetric</td>
    <td  align="center" class="tg-baqh">Asym.</td>
    <td  align="center" class="tg-baqh" colspan="2">Instance</td>
    <td  align="center" class="tg-baqh">Real</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh" colspan="2">Noise rate</td>
    <td  align="center" class="tg-baqh">0.2</td>
    <td  align="center" class="tg-baqh">0.4</td>
    <td  align="center" class="tg-baqh">0.6</td>
    <td  align="center" class="tg-baqh">0.4</td>
    <td  align="center" class="tg-baqh">0.2</td>
    <td  align="center" class="tg-baqh">0.4</td>
    <td  align="center" class="tg-baqh">noisy100</td>
    <!-- <td  align="center" class="tg-baqh">worst</td> -->
  </tr>
  <!-- baseline -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="4">baseline</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">63.94±0.39</td>
    <td  align="center" class="tg-baqh">52.39±0.63</td>
    <td  align="center" class="tg-baqh">41.20±0.25</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">45.60±0.17</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">62.54±0.50</td>
    <td  align="center" class="tg-baqh">44.48±0.15</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">54.55±0.13</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">67.98±0.13</td>
    <td  align="center" class="tg-baqh">61.86±0.11</td>
    <td  align="center" class="tg-baqh">49.32±0.65</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">52.99±0.51</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">66.14±0.16</td>
    <td  align="center" class="tg-baqh">53.51±0.16</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">60.43±0.07</td>
  </tr>
  <!-- PreResNet -->
  <tr>
    <td  align="center" class="tg-baqh">PreResNet</td>
    <td  align="center" class="tg-baqh">62.46±0.36</td>
    <td  align="center" class="tg-baqh">52.54±0.18</td>
    <td  align="center" class="tg-baqh">43.38±0.37</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">45.98±0.20</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">62.46±0.30</td>
    <td  align="center" class="tg-baqh">44.84±0.11</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">53.38±0.52</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">67.91±0.06</td>
    <td  align="center" class="tg-baqh">62.43±0.26</td>
    <td  align="center" class="tg-baqh">52.26±0.24</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">55.12±0.16</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">66.11±0.09</td>
    <td  align="center" class="tg-baqh">53.15±0.18</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">60.24±0.15</td>
  </tr>
  <!-- mixup -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="2">Mixup</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">66.33±0.26</td>
    <td  align="center" class="tg-baqh">58.74±0.49</td>
    <td  align="center" class="tg-baqh">48.89±0.77</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">52.32±0.36</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">62.28±0.55</td>
    <td  align="center" class="tg-baqh">44.41±0.67</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">58.34±0.29</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">71.08±0.31</td>
    <td  align="center" class="tg-baqh">64.86±0.08</td>
    <td  align="center" class="tg-baqh">55.23±0.40</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">58.59±0.15</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">67.75±0.07</td>
    <td  align="center" class="tg-baqh">51.10±0.13</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">63.17±0.22</td>
  </tr>
   <!-- labelsmoothing -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="2">Label Smoothing</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">65.81±0.54</td>
    <td  align="center" class="tg-baqh">53.61±0.32</td>
    <td  align="center" class="tg-baqh">41.86±0.71</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">47.91±0.09</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">62.82±0.36</td>
    <td  align="center" class="tg-baqh">41.62±0.44</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">55.71±0.44</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">68.73±0.14</td>
    <td  align="center" class="tg-baqh">62.38±0.59</td>
    <td  align="center" class="tg-baqh">49.93±0.48</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">57.31±0.27</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">66.90±0.12</td>
    <td  align="center" class="tg-baqh">52.68±0.35</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">61.13±0.27</td>
  </tr>
  <!-- Autoaug -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="2">Autoaug</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">68.44±0.04</td>
    <td  align="center" class="tg-baqh">59.10±0.35</td>
    <td  align="center" class="tg-baqh">48.16±0.59</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">53.53±0.45</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">66.63±0.09</td>
    <td  align="center" class="tg-baqh">46.48±0.48</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">58.30±0.25</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">71.80±0.17</td>
    <td  align="center" class="tg-baqh">66.03±0.17</td>
    <td  align="center" class="tg-baqh">55.74±0.38</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">60.30±0.29</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">68.03±0.04</td>
    <td  align="center" class="tg-baqh">48.78±0.13</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">63.49±0.09</td>
  </tr>
  <!-- Randaug -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="2">Randaug</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">66.98±0.13</td>
    <td  align="center" class="tg-baqh">57.78±0.28</td>
    <td  align="center" class="tg-baqh">46.82±0.47</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">52.39±0.05</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">65.17±0.13</td>
    <td  align="center" class="tg-baqh">44.62±0.37</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">57.11±0.07</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">71.11±0.08</td>
    <td  align="center" class="tg-baqh">64.67±0.21</td>
    <td  align="center" class="tg-baqh">53.95±0.35</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">59.45±0.29</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">66.88±0.08</td>
    <td  align="center" class="tg-baqh">47.47±0.45</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">62.98±0.23</td>
  </tr>
  <!-- Class-balance -->
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="2">Class Balance</td>
    <td  align="center" class="tg-baqh">Vanilla</td>
    <td  align="center" class="tg-baqh">64.15±0.16</td>
    <td  align="center" class="tg-baqh">52.74±0.59</td>
    <td  align="center" class="tg-baqh">41.03±0.35</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">47.87±0.33</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">65.12±0.20</td>
    <td  align="center" class="tg-baqh">53.42±0.13</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">55.28±0.15</td>
  </tr>
  <tr>
    <td  align="center" class="tg-baqh">EMA</td>
    <td  align="center" class="tg-baqh">67.43±0.08</td>
    <td  align="center" class="tg-baqh">61.34±0.34</td>
    <td  align="center" class="tg-baqh">48.83±0.52</td>
    <!-- asymmetric noise -->
    <td  align="center" class="tg-baqh">54.69±0.30</td>
    <!-- instance noise -->
    <td  align="center" class="tg-baqh">67.46±0.20</td>
    <td  align="center" class="tg-baqh">59.84±0.46</td>
    <!-- real noise -->
    <td  align="center" class="tg-baqh">60.50±0.21</td>
  </tr>
</tbody>
</table>
