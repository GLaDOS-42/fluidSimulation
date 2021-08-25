#include <iostream>
#include <pngwriter.h>
#include <pthread.h> 

using namespace std;

int WIDTH  = 1920 + 2;
int HEIGHT = 1080 + 2;
/*
// Thin and slow
int INFLOW_SIZE = 19;
float INFLOW_SPEED = 15;
float TIMESTEP = 0.1f;
int MAX_ITERATIONS = 42;
float PRESSURE_CONSTANT = 1;
int NUM_FRAMES = 6000;
*/
// Thick and fast
int INFLOW_SIZE = 42;
float INFLOW_SPEED = 69;
float TIMESTEP = 0.05f;
int MAX_ITERATIONS = 42;
float PRESSURE_CONSTANT = 1;
int NUM_FRAMES = 3600;

float POOL_COLOR[] = { 0.0, 1.0, 1.0 };
float JET_COLOR[] = { 1.0, 0.5, 0.0 };

int numThreads = 16;

//very crude threshold normalization
float basicNormalize(float value)
{
	if(value > 1.0f) { return 1.0f; }
	return value;
}

//improved normalization with sigmoid function
float sigmoid(float value)
{
	return (float)(1 / (1 + exp((double)value)));
}

//faster approximation of sigmoid - range from -1 to 1s
float fastSigmoid(float value)
{
	return value / (1 + abs(value));
}

//linear interpolation for 0<x<1
//a and b are values above and below x, respectively
inline float lerp(float a, float b, float x)
{
	return a * (1.0f - x) + b * x;
}    

inline float cubicInterpolate(float x0, float x1, float x2, float x3, float x)
{
	float x1prime = 0.5f * (x2 - x0);
	float x2prime = 0.5f * (x3 - x1);
	float a =  2 * x1 +     x1prime - 2 * x2 + x2prime;
	float b = -3 * x1 - 2 * x1prime + 3 * x2 - x2prime;
	float c = x1prime;
	float d = x1;
	return a * (x * x * x) + b * (x * x) + c * (x) + d;
}

//container for a MAC grid quantity
//cells at the edges are not considered when computing advection, pressure gradients, etc
//they are "ghost cells" to enforce boundary conditions
class FluidQuantity
{
    public:
    
    //source and destination buffers
    float *src;
    float *dst;
	
	int w, h;
	float offsetX, offsetY;

	float timestep;
    
    FluidQuantity(int width, int height, float ox, float oy, float timestep)
    {
		this->w = width;
		this->h = height;
		this->offsetX = ox;
		this->offsetY = oy;
		this->timestep = timestep;
		src = (float*) calloc(w * h, sizeof(float));
        dst = (float*) calloc(w * h, sizeof(float));
		for(int i = 0; i != w; i++)
        {
            for(int j = 0; j != h; j++)
            {
				src[getPosition(i, j)] = 0.0f;
            }
        }
    }
	
	//compute position in buffer
	inline int getPosition(int i, int j)
	{
		return i + w * j;
	}
	
	inline float getSrc(int i, int j)
	{
		return src[getPosition(i, j)];
	}
	
	//swap the source and destination buffers
	void swap()
	{
		float *tmp = src;
		src = dst;
		dst = tmp;
	}
    
    //bilinear interpolation
    //coordinates are constrained to lie within boundary
    float bilerp(float x, float y)
    {
        //clamping coordinates
        // if(x < 0) { x = 0.001f; }
        // if(y < 0) { y = 0.001f; }
        // if(x > w) { x = w - 0.001f; }
        // if(y > h) { y = h - 0.001f; }
        
        //distances from nearby points
        int intX = (int)x, intY = (int)y;
        x -= intX;
        y -= intY;
        
        //values at nearby grid points
        float x00 = getSrc(intX, intY);
        float x10 = getSrc(intX + 1, intY);
        float x01 = getSrc(intX, intY + 1);
        float x11 = getSrc(intX + 1, intY + 1);
        
        //interpolate
        return lerp(lerp(x00, x10, x), lerp(x01, x11, x), y);
    }

	//bicubic interpolation
	//smoother than bilinear, may improve advection where there is a sharp gradient
	float bicubic(float xPos, float yPos)
	{
		float x = xPos;
		float y = yPos;
		
		//clamping coordinates
        if(x < 3) { x = 3.001f; }
        if(y < 3) { y = 3.001f; }
        if(x > w - 3) { x = w - 3.001f; }
        if(y > h - 3) { y = h - 3.001f; }

		//distances from nearby points
        int intX = (int)x, intY = (int)y;
        x -= intX;
        y -= intY;

		//values at nearby grid points
        float  x00 = getSrc(intX - 1, intY - 1);
        float  x01 = getSrc(intX - 1, intY);
		float  x02 = getSrc(intX - 1, intY + 1);
		float  x03 = getSrc(intX - 1, intY + 2);
		float  x10 = getSrc(intX,     intY - 1);
		float  x11 = getSrc(intX,     intY);
		float  x12 = getSrc(intX,     intY + 1);
		float  x13 = getSrc(intX,     intY + 2);
		float  x20 = getSrc(intX + 1, intY - 1);
		float  x21 = getSrc(intX + 1, intY);
		float  x22 = getSrc(intX + 1, intY + 1);
		float  x23 = getSrc(intX + 1, intY + 2);
		float  x30 = getSrc(intX + 2, intY - 1);
		float  x31 = getSrc(intX + 2, intY);
		float  x32 = getSrc(intX + 2, intY + 1);
		float  x33 = getSrc(intX + 2, intY + 2);

		//intermediate values
		float v0 = cubicInterpolate(x00, x10, x20, x30, x);
		float v1 = cubicInterpolate(x01, x11, x21, x31, x);
		float v2 = cubicInterpolate(x02, x12, x22, x32, x);
		float v3 = cubicInterpolate(x03, x13, x23, x33, x);

		//interpolate
		return cubicInterpolate(v0, v1, v2, v3, y);
	}
	
	//simple euler's method for tracing particle position back in time
	float euler(int x, int y, FluidQuantity *u, FluidQuantity *v)
	{
		float xPos = (float)x, yPos = (float)y;
		xPos -= timestep * u->getSrc(x, y);
		yPos -= timestep * v->getSrc(x, y);
        return bicubic(xPos, yPos);
	}

	//semi-Lagrangian advection
	void advect(FluidQuantity *u, FluidQuantity *v)
	{
		for(int i = 1; i < w - 1; i++)
		{
			for(int j = 1; j < h - 1; j++)
			{
				dst[getPosition(i, j)] = euler(i, j, u, v);
				dst[getPosition(i, j)] = euler(i, j, u, v);
			}
		}
	}
	
	//gradient in x direction
	//this needs to be read from the destination buffer
	float gradientX(int x, int y)
	{
		return (dst[getPosition(x + 1, y)] - dst[getPosition(x - 1, y)]) / 2.0f;
	}
	
	//gradient in y direction
	float gradientY(int x, int y)
	{
		return (dst[getPosition(x, y + 1)] - dst[getPosition(x, y - 1)]) / 2.0f;
	}
	
	//sets quantity inside a box to the given value
	void addInflow(int x1, int x2, int y1, int y2, float value)
	{
		for(int i = x1; i != x2; i++)
		{
			for(int j = y1; j != y2; j++)
			{
				src[getPosition(i, j)] = value;
			}
		}
	}

	void addInflowDiag(int x, int y, int width, int length, float value) 
	{
		for (int i = 0; i < width; i++) {
			int xd = 1, yd = 0;
			for (int j = 0; j < length; j++) {
				src[getPosition(x + i + j, y - i + j)] = value;
				if (i < width - 1) src[getPosition(x + i + j, y - i + j - 1)] = value;
			}
			for (int j = 0; j < width - i; j++) {
				src[getPosition(x + i, y - i - j)] = value;
			}
		}
	}
};

class FluidSolver
{
	public:

	FluidQuantity *u;
	FluidQuantity *v;
	FluidQuantity *pressure;
	FluidQuantity *dyeAmount;
    int w;
	int h;
	float timeStep;
    
	pthread_mutex_t work_lock; 
	pthread_cond_t work_cond;   
	int pending; 
	int complete; 
  
    FluidSolver(int width, int height)
    {
		w = width;
		h = height;
		u = new FluidQuantity(w, h, 0.0f, 0.0f, TIMESTEP);
		v = new FluidQuantity(w, h, 0.0f, 0.0f, TIMESTEP);
		pressure  = new FluidQuantity(w, h, 0.5f, 0.5f, TIMESTEP);
		dyeAmount = new FluidQuantity(w, h, 0.0f, 0.0f, TIMESTEP);
		timeStep = TIMESTEP;
		pthread_mutex_init(&work_lock, NULL); 
		pthread_cond_init(&work_cond, NULL); 
		pending = 0; 
		complete = numThreads; 
    }
	
	//divergence of intermediate velocity field
	//should be 0 - hence the pressure correction
	inline float divergence(int x, int y)
	{
		float dx = (u->dst[u->getPosition(x + 1, y)] - u->dst[u->getPosition(x - 1, y)]) / 2.0f;
		float dy = (v->dst[v->getPosition(x, y + 1)] - v->dst[v->getPosition(x, y - 1)]) / 2.0f;
		return dx + dy;
	}
	
	//curl of velocity field - calculated after pressure correction
	inline float curl(int x, int y)
	{
		float dvx = (v->dst[v->getPosition(x + 1, y)] - v->dst[v->getPosition(x - 1, y)]) / 2.0f;
		float duy = (u->dst[u->getPosition(x, y + 1)] - u->dst[u->getPosition(x, y - 1)]) / 2.0f;
		return dvx - duy;
	}

	//jacobi solver for pressure correction
	//can be improved by estimating error, instead of executing a fixed number of iteration
	void jacobi()
	{
		pthread_mutex_lock(&work_lock);
		for(int n = 0; n != MAX_ITERATIONS; n++)
		{
			while (complete < numThreads) pthread_cond_wait(&work_cond, &work_lock);
			pending = numThreads;
			complete = 0;
			pthread_mutex_unlock(&work_lock);
			pthread_cond_broadcast(&work_cond);
			pthread_mutex_lock(&work_lock);
			while (complete < numThreads) pthread_cond_wait(&work_cond, &work_lock);
			pressure->swap();
			setPressureBoundaryConditions();
		}
		pthread_mutex_unlock(&work_lock);
	}

	void jacobi_part(int offset, int modulo) 
	{
		int chunk = (int)ceil((double)h / (double)modulo);
		int min = offset * chunk;
		int max = min + chunk;
		if (min < 1) min = 1;
		if (max > h - 1) max = h - 1;
		for(int i = 1; i < w - 1; i++)
		{
			for(int j = min; j < max; j++)
			{
				float p1 = pressure->getSrc(i, j + 1);
				float p2 = pressure->getSrc(i, j - 1);
				float p3 = pressure->getSrc(i + 1, j);
				float p4 = pressure->getSrc(i - 1, j);
				float div = divergence(i, j) / (PRESSURE_CONSTANT);
				pressure->dst[pressure->getPosition(i, j)] = (p1 + p2 + p3 + p4 - div) / 4.0f;
			}
		}
	}

	//subtract pressure gradient from intermediate velocity field
	void applyPressure()
	{
		for(int i = 1; i != w - 1; i++)
		{
			for(int j = 1; j != h - 1; j++)
			{
				u->dst[u->getPosition(i, j)] -= PRESSURE_CONSTANT * pressure->gradientX(i, j);
				v->dst[v->getPosition(i, j)] -= PRESSURE_CONSTANT * pressure->gradientY(i, j);
			}
		}
	}

	//set "ghost cell" boundary conditions
	void setBoundaryConditions()
	{
		for(int i = 1; i < w - 1; i++)
		{
			v->src[v->getPosition(i, 0)]   = -(v->src[v->getPosition(i,   1)]);
			v->src[v->getPosition(i, h-1)] = -(v->src[v->getPosition(i, h-2)]);
			u->src[u->getPosition(i, 0)]   =  (u->src[u->getPosition(i,   1)]);
			u->src[u->getPosition(i, h-1)] =  (u->src[u->getPosition(i, h-2)]);
			// dyeAmount->src[dyeAmount->getPosition(i, 0)]   = (dyeAmount->src[dyeAmount->getPosition(i,   1)]);
			// dyeAmount->src[dyeAmount->getPosition(i, h-1)] = (dyeAmount->src[dyeAmount->getPosition(i, h-2)]);
		}

		for(int j = 1; j < h - 1; j++)
		{
			u->src[u->getPosition(0, j)]   = -(u->src[u->getPosition(1,   j)]);
			u->src[u->getPosition(w-1, j)] = -(u->src[u->getPosition(w-2, j)]);
			v->src[v->getPosition(0, j)]   =  (v->src[v->getPosition(1,   j)]);
			v->src[v->getPosition(w-1, j)] =  (v->src[v->getPosition(w-2, j)]);
			// dyeAmount->src[dyeAmount->getPosition(0, j)]   = (dyeAmount->src[dyeAmount->getPosition(1,   j)]);
			// dyeAmount->src[dyeAmount->getPosition(w-1, j)] = (dyeAmount->src[dyeAmount->getPosition(w-2, j)]);
		}

		setPressureBoundaryConditions();
	}

	//set pressure in boundary cells, for use in jacobi iterations
	void setPressureBoundaryConditions()
	{
		for(int i = 1; i < w - 1; i++)
		{
			pressure->src[pressure->getPosition(i, 0)]   = pressure->src[pressure->getPosition(i,   1)];
			pressure->src[pressure->getPosition(i, h-1)] = pressure->src[pressure->getPosition(i, h-2)];
		}

		for(int j = 1; j < h - 1; j++)
		{
			pressure->src[pressure->getPosition(0, j)]   = pressure->src[pressure->getPosition(1,   j)];
			pressure->src[pressure->getPosition(w-1, j)] = pressure->src[pressure->getPosition(w-2, j)];
		}
	}

	//update the fluid domain through one timestep
	void update()
	{
		    u->addInflow(0, INFLOW_SIZE, 0, INFLOW_SIZE, INFLOW_SPEED); // in m/sec
		    v->addInflow(0, INFLOW_SIZE, 0, INFLOW_SIZE, INFLOW_SPEED); // in m/sec

			u->addInflow(w - INFLOW_SIZE, w, 0, INFLOW_SIZE, -1 * INFLOW_SPEED); // in m/sec
		    v->addInflow(w - INFLOW_SIZE, w, 0, INFLOW_SIZE, INFLOW_SPEED); // in m/sec

			u->addInflow(0, INFLOW_SIZE, h - INFLOW_SIZE, h, INFLOW_SPEED); // in m/sec
		    v->addInflow(0, INFLOW_SIZE, h - INFLOW_SIZE, h, -1 * INFLOW_SPEED); // in m/sec

			u->addInflow(w - INFLOW_SIZE, w, h - INFLOW_SIZE, h, -1 * INFLOW_SPEED); // in m/sec
		    v->addInflow(w - INFLOW_SIZE, w, h - INFLOW_SIZE, h, -1 * INFLOW_SPEED); // in m/sec

			// dyeAmount->addInflow(0, INFLOW_SIZE, 0, INFLOW_SIZE, 1);
		//         u->addInflowDiag(0, INFLOW_SIZE - 1, INFLOW_SIZE, 1.2 * INFLOW_SIZE, INFLOW_SPEED); // in m/sec
		//         v->addInflowDiag(0, INFLOW_SIZE - 1, INFLOW_SIZE, 1.2 * INFLOW_SIZE, INFLOW_SPEED); // in m/sec
		// dyeAmount->addInflowDiag(0, INFLOW_SIZE - 1, INFLOW_SIZE, 1.2 * INFLOW_SIZE, 1);
		u->advect(u, v);
		v->advect(u, v);
		//dyeAmount->advect(u, v);
		setBoundaryConditions();
		jacobi();
		applyPressure();
		setBoundaryConditions();
	}

	void swapAll()
	{
		u->swap();
		v->swap();
		pressure->swap();
		dyeAmount->swap();
	}
};

void blendColor(float col1[], float col2[], float ratio, float res[]) 
{
	res[0] = lerp(col1[0], col2[0], ratio);
	res[1] = lerp(col1[1], col2[1], ratio);
	res[2] = lerp(col1[2], col2[2], ratio);
}

void blendColorMul(float col1[], float col2[], float ratio, float res[]) 
{
	res[0] = col1[0] * (1 - col2[0] * ratio);
	res[1] = col1[1] * (1 - col2[1] * ratio);
	res[2] = col1[2] * (1 - col2[2] * ratio);
	// res[1] = lerp(col1[1], col2[1], ratio);
	// res[2] = lerp(col1[2], col2[2], ratio);
}

void blendTwoColorsBlack(float col1[1], float col2[], float value, float res[])
{
	if(value >= 0.1)
	{
		res[0] = col1[0] * abs(value);
		res[1] = col1[1] * abs(value);
		res[2] = col1[2] * abs(value);
	}
	else if(value < -0.1f)
	{
		res[0] = abs(col2[0] * abs(value));
		res[1] = abs(col2[1] * abs(value));
		res[2] = abs(col2[2] * abs(value));
	}
	else
	{
		res[0] = 0, res[1] = 0, res[2] = 0;
	}	
}

typedef struct worker_cfg_t
{
	FluidSolver *solver;
	int          modulo;
	int          offset;
} worker_cfg_t;


void *worker(void *vargp) 
{
	worker_cfg_t *worker_cfg = (worker_cfg_t*)vargp;
	while (true) {
	    pthread_mutex_lock(&worker_cfg->solver->work_lock);
		while (worker_cfg->solver->pending == 0) {
			// nothing to do
			pthread_cond_wait(&worker_cfg->solver->work_cond, &worker_cfg->solver->work_lock);
		}
		worker_cfg->solver->pending--;
		// we've got work!
		pthread_mutex_unlock(&worker_cfg->solver->work_lock);
		pthread_cond_broadcast(&worker_cfg->solver->work_cond);
		worker_cfg->solver->jacobi_part(worker_cfg->offset, worker_cfg->modulo);
	    pthread_mutex_lock(&worker_cfg->solver->work_lock);
		worker_cfg->solver->complete++;
		pthread_mutex_unlock(&worker_cfg->solver->work_lock);
		pthread_cond_broadcast(&worker_cfg->solver->work_cond);
	}
    return NULL;
} 
   
int main()
{
	FluidSolver *solver = new FluidSolver(WIDTH, HEIGHT);
	char frameName[1024];
	float color[3];

	pthread_t *threads = (pthread_t*)calloc(numThreads, sizeof(pthread_t));
	worker_cfg_t *worker_cfg = (worker_cfg_t*)calloc(numThreads, sizeof(worker_cfg_t));
	for(int i = 0; i < numThreads; i++) {
		worker_cfg[i].solver = solver;
		worker_cfg[i].modulo = numThreads;
		worker_cfg[i].offset = i;
    	pthread_create(&threads[i], NULL, worker, &worker_cfg[i]); 
	}

	for(int frame = 0; frame != NUM_FRAMES; frame++)
	{
		solver->update();
		if(frame % 2 == 0)
		{
			sprintf(frameName, "output/frame-%04d.png", frame / 2);
		}
		pngwriter image(WIDTH - 2, HEIGHT - 2, 0.0, frameName);
		for(int i = 0; i < WIDTH - 2; i++)
		{
			for(int j = 0; j < HEIGHT - 2; j++)
			{
				// float colorScale = solver->dyeAmount->dst[solver->dyeAmount->getPosition(i+1, j+1)];
				// float U = solver->u->dst[solver->u->getPosition(i+1, j+1)];
				// float V = solver->v->dst[solver->v->getPosition(i+1, j+1)];
				// float colorScale = basicNormalize(U * U + V * V);
				// blendColor(POOL_COLOR, JET_COLOR, colorScale, color);
				float colorScale = fastSigmoid(solver->curl(i+1, j+1));
				// color[0] = colorScale, color[1] = colorScale, color[2] = colorScale;
				blendTwoColorsBlack(POOL_COLOR, JET_COLOR, colorScale, color);
				image.plot(i, j, (double)color[0], (double)color[1], (double)color[2]);

			}
		}
		image.close();
		solver->swapAll();
	}

    return 0;
}
