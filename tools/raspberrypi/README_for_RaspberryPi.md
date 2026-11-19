# 라즈베리 파이 세팅

hostname : rad2pi

usernamer : radar

password : radar1234

window powershell에서) 

```tsx
ssh radar@rad2pi.local
```

입력하면 라즈베리파이에 연결됨

비밀번호 입력을 요청하면 radar1234 입력하면 됨

PS C:\...> : Windows PC에서 명령어 실행

radar@rad2pi:~ $ : Raspberry Pi에서 명령어 실행

라즈베리 파이 종료

```tsx
sudo shutdown -h now
```

→ 이 명령어 입력해도 라즈베리 파이에 빨간불(전원어댑터 연결)은 켜는데 신경 쓰지 말고 그냥 어댑터 뽑으면 됨

중간중간 raspberry pi 종료하지는 않고 연결 잠깐 끊는 법

```tsx
exit
```

---

## Raspberry Pi 에서 카메라 촬영

**현재 라즈베리파이와 연결된 카메라 목록 출력**

```tsx
rpicam-hello --list-cameras
```

imx708 센서가 연결되어있어야 함

사진 촬영

```tsx
rpicam-still -n -t 2000 -o test.jpg
// 현재 디렉토리가 사용자 홈(~)이라면 저장 경로 /home/radar/test.jpg

// 저장 위치를 명확하게 지정하고 싶다면 
rpicam-still -n -t 2000 -o /home/radar/test.jpg
```

촬영된 파일 확인

```tsx
ls -lh ~/test.jpg
```

Window로 사진 가져오는 법

```tsx
1. terminal에서 exit 명령어 입력
   (window powershell 프롬프트 PS C:\~~~ 이렇게 나오는지 확인)
   
2. scp radar@rad2pi.local:~/test.jpg .
   위 명령어 입력하면 test.jpg를 windows powershell의 현재 폴더로 복사함. 
```