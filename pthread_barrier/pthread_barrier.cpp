#include<iostream>
#include<Windows.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pthread.h>
#include<immintrin.h>
#include<semaphore.h>//信号量包含的头文件
using namespace std;
alignas(16) float gdata[10000][10000];//进行对齐操作
float gdata2[10000][10000];
float gdata1[10000][10000];
float gdata3[10000][10000];
int n;
//定义线程数据结构
typedef struct {
	int t_id;	//线程编号
}threadParam_t;
const int Num_thread = 8;//8核CPU

pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

//线程函数1
void* threadFunc_SSE(void* Param)//利用void类型的指针可指向任意类型的数据
{
	threadParam_t* p = (threadParam_t*)Param;//强制类型转换成线程数据结构
	int t_id = p->t_id;//线程的编号获取
	//让一个线程负责除法操作，其运算完后也会进行后续消元操作
	for (int k = 0; k < n; k++)
	{
		__m128 r0, r1, r2, r3;//四路运算，定义四个float向量寄存器
		if (t_id == 0)
		{
			float temp[4] = { gdata3[k][k],gdata3[k][k],gdata3[k][k],gdata3[k][k] };
			r0 = _mm_loadu_ps(temp);//内存不对齐的形式加载到向量寄存器中
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = _mm_loadu_ps(gdata3[k] + j);
				r1 = _mm_div_ps(r1, r0);//将两个向量对位相除
				_mm_storeu_ps(gdata3[k], r1);//相除结果重新放回内存
			}
			//对剩余不足4个的数据进行消元
			for (j; j < n; j++)
			{
				gdata3[k][j] = gdata3[k][j] / gdata3[k][k];
			}
			gdata3[k][k] = 1.0;
		}
		pthread_barrier_wait(&barrier_Division);//在进行除法操作后使用barrier直接线程同步

		//任务划分，以行为单位进行划分

		for (int i = k + 1 + t_id; i < n; i += Num_thread)
		{
			//划分完对三重循环结合SIMD运算
			float temp2[4] = { gdata3[i][k],gdata3[i][k],gdata3[i][k],gdata3[i][k] };
			r0 = _mm_loadu_ps(temp2);
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = _mm_loadu_ps(gdata3[k] + j);
				r2 = _mm_loadu_ps(gdata3[i] + j);
				r3 = _mm_mul_ps(r0, r1);
				r2 = _mm_sub_ps(r2, r3);
				_mm_storeu_ps(gdata3[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata3[i][j] = gdata3[i][j] - (gdata3[i][k] * gdata3[k][j]);
			}
			gdata3[i][k] = 0;
		}
		pthread_barrier_wait(&barrier_Elimination);//在消元操作后再次进行线程同步
	}
	pthread_exit(NULL);
	return NULL;
}

//线程函数2
void* threadFunc_AVX(void* Param)//利用void类型的指针可指向任意类型的数据
{
	threadParam_t* p = (threadParam_t*)Param;//强制类型转换成线程数据结构
	int t_id = p->t_id;//线程的编号获取
	//让一个线程负责除法操作，其运算完后也会进行后续消元操作
	for (int k = 0; k < n; k++)
	{
		__m256 r0, r1, r2, r3;//四路运算，定义四个float向量寄存器
		if (t_id == 0)
		{
			float temp[8] = { gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k] };
			r0 = _mm256_load_ps(temp);//内存对齐的形式加载到向量寄存器中
			int j;
			for (j = k + 1; j + 8 <= n; j += 8)
			{
				r1 = _mm256_load_ps(gdata2[k] + j);
				r1 = _mm256_div_ps(r1, r0);//将两个向量对位相除
				_mm256_store_ps(gdata2[k], r1);//相除结果重新放回内存
			}
			//对剩余不足8个的数据进行消元
			for (j; j < n; j++)
			{
				gdata2[k][j] = gdata2[k][j] / gdata2[k][k];
			}
			gdata2[k][k] = 1.0;
		}

		pthread_barrier_wait(&barrier_Division);
		//任务划分，以行为单位进行划分

		for (int i = k + 1 + t_id; i < n; i += Num_thread)
		{
			//划分完对三重循环结合SIMD运算
			float temp2[8] = { gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k] };
			r0 = _mm256_load_ps(temp2);
			int j;
			for (j = k + 1; j + 8 <= n; j += 8)
			{
				r1 = _mm256_load_ps(gdata2[k] + j);
				r2 = _mm256_load_ps(gdata2[i] + j);
				r3 = _mm256_mul_ps(r0, r1);
				r2 = _mm256_sub_ps(r2, r3);
				_mm256_store_ps(gdata2[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata2[i][j] = gdata2[i][j] - (gdata2[i][k] * gdata2[k][j]);
			}
			gdata2[i][k] = 0;
		}
		pthread_barrier_wait(&barrier_Elimination);
	}
	pthread_exit(NULL);
	return NULL;
}

//数据初始化
void Initialize(int N)
{
	for (int i = 0; i < N; i++)
	{
		//首先将全部元素置为0，对角线元素置为1
		for (int j = 0; j < N; j++)
		{
			gdata[i][j] = 0;
			gdata1[i][j] = 0;
			gdata2[i][j] = 0;
			gdata3[i][j] = 0;
		}
		gdata[i][i] = 1.0;
		//将上三角的位置初始化为随机数
		for (int j = i + 1; j < N; j++)
		{
			gdata[i][j] = rand();
			gdata1[i][j] = gdata[i][j] = gdata2[i][j] = gdata3[i][j];
		}
	}
	for (int k = 0; k < N; k++)
	{
		for (int i = k + 1; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				gdata[i][j] += gdata[k][j];
				gdata1[i][j] += gdata1[k][j];
				gdata2[i][j] += gdata2[k][j];
				gdata3[i][j] += gdata3[k][j];
			}
		}
	}

}
//平凡算法
void Normal_alg(int N)
{
	int i, j, k;
	for (k = 0; k < N; k++)
	{
		for (j = k + 1; j < N; j++)
		{
			gdata1[k][j] = gdata1[k][j] / gdata1[k][k];
		}
		gdata1[k][k] = 1.0;
		for (i = k + 1; i < N; i++)
		{
			for (j = k + 1; j < N; j++)
			{
				gdata1[i][j] = gdata1[i][j] - (gdata1[i][k] * gdata1[k][j]);
			}
			gdata1[i][k] = 0;
		}
	}
}

//对全部进行SIMD优化
void Par_alg_all(int n)
{
	int i, j, k;
	__m128 r0, r1, r2, r3;//四路运算，定义四个float向量寄存器
	for (k = 0; k < n; k++)
	{
		float temp[4] = { gdata[k][k],gdata[k][k],gdata[k][k],gdata[k][k] };
		r0 = _mm_loadu_ps(temp);//内存不对齐的形式加载到向量寄存器中
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			r1 = _mm_loadu_ps(gdata[k] + j);
			r1 = _mm_div_ps(r1, r0);//将两个向量对位相除
			_mm_storeu_ps(gdata[k], r1);//相除结果重新放回内存
		}
		//对剩余不足4个的数据进行消元
		for (j; j < n; j++)
		{
			gdata[k][j] = gdata[k][j] / gdata[k][k];
		}
		gdata[k][k] = 1.0;
		//以上对应上述第一个二重循环优化的SIMD

		for (i = k + 1; i < n; i++)
		{
			float temp2[4] = { gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k] };
			r0 = _mm_loadu_ps(temp2);
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = _mm_loadu_ps(gdata[k] + j);
				r2 = _mm_loadu_ps(gdata[i] + j);
				r3 = _mm_mul_ps(r0, r1);
				r2 = _mm_sub_ps(r2, r3);
				_mm_storeu_ps(gdata[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata[i][j] = gdata[i][j] - (gdata[i][k] * gdata[k][j]);
			}
			gdata[i][k] = 0;
		}
	}
}

void pthread_SSE()
{
	//初始化barrier
	pthread_barrier_init(&barrier_Division, NULL, Num_thread);
	pthread_barrier_init(&barrier_Elimination, NULL, Num_thread);
	pthread_t* handles = new pthread_t[Num_thread];//创建线程句柄
	threadParam_t* param = new threadParam_t[Num_thread];//需要传递的参数打包
	for (int t_id = 0; t_id < Num_thread; t_id++)
	{
		param[t_id].t_id = t_id;//将线程参数传递（线程名）
		pthread_create(&handles[t_id], NULL, threadFunc_SSE, &param[t_id]);//创建线程函数
	}
	for (int t_id = 0; t_id < Num_thread; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&barrier_Division);
	pthread_barrier_destroy(&barrier_Elimination);
}

void pthread_AVX()
{
	//初始化barrier
	pthread_barrier_init(&barrier_Division, NULL, Num_thread);
	pthread_barrier_init(&barrier_Elimination, NULL, Num_thread);
	pthread_t* handles = new pthread_t[Num_thread];//创建线程句柄
	threadParam_t* param = new threadParam_t[Num_thread];//需要传递的参数打包
	for (int t_id = 0; t_id < Num_thread; t_id++)
	{
		param[t_id].t_id = t_id;//将线程参数传递（线程名）
		pthread_create(&handles[t_id], NULL, threadFunc_AVX, &param[t_id]);//创建线程函数
	}
	for (int t_id = 0; t_id < Num_thread; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&barrier_Division);
	pthread_barrier_destroy(&barrier_Elimination);
}


int main()
{
	LARGE_INTEGER fre, begin, end;
	double gettime;
	cin >> n;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Initialize(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "intial time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Normal_alg(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "normal time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	pthread_SSE();
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "pthread_SSE time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	pthread_AVX();
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "pthread_AVX time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Par_alg_all(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "SIMD time: " << gettime << " ms" << endl;
}
