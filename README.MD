### 训练

```python
python main.py --attack_mode none
python main.py --attack_mode pgd
python main.py --attack_mode free
python main.py --attack_mode fgsm
```

### 测试
```python
python main.py --test --attack_mode none
python main.py --test --attack_mode pgd
python main.py --test --attack_mode free
python main.py --test --attack_mode fgsm
```