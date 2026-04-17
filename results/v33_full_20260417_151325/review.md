# V33 Manual Review (61 samples)

> Run: v33_full_20260417_151325
> Model: Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM 0.19.0
> Pipeline: vllm_pipeline/run_vllm.py (20 POST hooks)
> Automated P2 scan: 0 P2 (17/17 V32 P2s verified fixed, 0 regressions)
> Status: **审查中 — 5/61 逐字审查完成，P2=0。下一个：ROW 7（line 446）**

## 汇总统计

| 严重度 | 数量 | 比率 |
|--------|------|------|
| **P0** | 0 | 0% |
| **P1** | 0 | 0% |
| **P2** | 1 | — |

## 审查记录

| ROW | coral_idx | P2 | 备注 |
|-----|-----------|-----|------|
| 1 | 140 | 0 | ✅✅ mBC lungs/liver/peritoneum/ovary (no brain!), ibrance+unspecified conditional, bone scan+MRI brain+6 labs, Integrative Med, full code (逐字审查) |
| 2 | 141 | 0 | ✅✅ mTNBC+Lynch, irinotecan dose change 150mg/m2 q2w, pRBC+NS+KCl, Na124/K3.1/Hgb7.7, doxycycline, HBV q4mo, 30+labs (逐字审查) |
| 3 | 142 | 0 | ✅✅ Stage IIA IDC 1.7cm node+, HR+/HER2-(FISH neg), Ki-67 30-35%, 2nd opinion, PET+genetics pending, full code (逐字审查) |
| 5 | 144 | 0 | ✅✅ Stage III→IV, DM correctly "bone (sternum)" only, regional LNs in Metastasis not DM, mixed response, anastrozole+palbo+leuprolide (逐字审查) |
| 6 | 145 | 0 | ✅✅ G1 IDC 1.5cm MammaPrint Low, zoladex+letrozole, bipolar 2, Myriad neg, estradiol monthly, comprehensive labs (逐字审查) |
| 7 | 146 | | |
| 8 | 147 | | |
| 9 | 148 | | |
| 10 | 149 | | |
| 11 | 150 | | |
| 12 | 151 | | |
| 14 | 153 | | |
| 17 | 156 | | |
| 18 | 157 | | |
| 20 | 159 | | |
| 22 | 161 | | |
| 27 | 166 | | |
| 29 | 168 | | |
| 30 | 169 | | |
| 33 | 172 | | |
| 34 | 173 | | |
| 36 | 175 | | |
| 37 | 176 | | |
| 40 | 179 | | |
| 41 | 180 | | |
| 42 | 181 | | |
| 43 | 182 | | |
| 44 | 183 | | |
| 46 | 185 | | |
| 49 | 188 | | |
| 50 | 189 | | |
| 52 | 191 | | |
| 53 | 192 | | |
| 54 | 193 | | |
| 57 | 196 | | |
| 59 | 198 | | |
| 61 | 200 | | |
| 63 | 202 | | |
| 64 | 203 | | |
| 65 | 204 | | |
| 66 | 205 | | |
| 68 | 207 | | |
| 70 | 209 | | |
| 72 | 211 | | |
| 73 | 212 | | |
| 78 | 217 | | |
| 80 | 219 | | |
| 82 | 221 | | |
| 83 | 222 | | |
| 84 | 223 | | |
| 85 | 224 | | |
| 86 | 225 | | |
| 87 | 226 | | |
| 88 | 227 | | |
| 90 | 229 | | |
| 91 | 230 | | |
| 92 | 231 | | |
| 94 | 233 | | |
| 95 | 234 | | |
| 97 | 236 | | |
| 100 | 239 | | |
