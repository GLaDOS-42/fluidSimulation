#include <iostream>
#include <pngwriter.h>
using namespace std;

int WIDTH  = 480;
int HEIGHT = 480;
float TIMESTEP = 0.1f;
int MAX_ITERATIONS = 42;
float PRESSURE_CONSTANT = 1;
int NUM_FRAMES = 4000;

float basicNormalize(float value)
{
	if(value > 1.0f){ return 1.0f; }
	return value;
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
	inline int getPosition(float i, float j)
	{
		return (int)i + w * (int)j;
	}
	
	float getSrc(int i, int j)
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
    
    //linear interpolation for 0<x<1
    //a and b are values above and below x, respectively
    float lerp(float a, float b, float x)
    {
        return a * (1.0f - x) + b * x;
    }
    
    //bilinear interpolation
    //coordinates are constrained to lie within boundary
    float bilerp(float x, float y)
    {
        //clamping coordinates
        if(x < 0) { x = 0.001f; }
        if(y < 0) { y = 0.001f; }
        if(x > w) { x = w - 1.001f; }
        if(y > h) { y = h - 1.001f; }
        
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
	
	//simple euler's method for tracing particle position back in time
	float euler(int x, int y, FluidQuantity *u, FluidQuantity *v)
	{
		float xPos = (float)x, yPos = (float)y;
		xPos -= timestep * u->getSrc(x, y);
		yPos -= timestep * v->getSrc(x, y);
        return bilerp(xPos, yPos);
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
};

class FluidSolver
{
	public:

	FluidQuantity *u;
	FluidQuantity *v;
	FluidQuantity *pressure;
	FluidQuantity *dyeAmount;
    int w = WIDTH;
	int h = HEIGHT;
	float timeStep;
    
    FluidSolver(int width, int height)
    {
		w = width;
		h = height;
		u = new FluidQuantity(w, h, 0.0f, 0.0f, TIMESTEP);
		v = new FluidQuantity(w, h, 0.0f, 0.0f, TIMESTEP);
		pressure  = new FluidQuantity(w, h, 0.5f, 0.5f, TIMESTEP);
		dyeAmount = new FluidQuantity(w, h, 0.0f, 0.0f, TIMESTEP);
		timeStep = TIMESTEP;
    }
	
	//divergence of intermediate velocity field
	//should be 0 - hence the pressure correction
	float divergence(int x, int y)
	{
		float dx = (u->dst[u->getPosition(x + 1, y)] - u->dst[u->getPosition(x - 1, y)]) / 2.0f;
		float dy = (v->dst[v->getPosition(x, y + 1)] - v->dst[v->getPosition(x, y - 1)]) / 2.0f;
		return dx + dy;
	}
	
	//jacobi solver for pressure correction
	//can be improved by estimating error, instead of executing a fixed number of iteration
	void jacobi()
	{
		for(int n = 0; n != MAX_ITERATIONS; n++)
		{
			for(int i = 1; i != w - 1; i++)
			{
				for(int j = 1; j != h - 1; j++)
				{
					float p1 = pressure->getSrc(i, j + 1);
					float p2 = pressure->getSrc(i, j - 1);
					float p3 = pressure->getSrc(i + 1, j);
					float p4 = pressure->getSrc(i - 1, j);
					float div = divergence(i, j);
					pressure->dst[pressure->getPosition(i, j)] = (p1 + p2 + p3 + p4 - div) / 4.0f;
				}
			}
			pressure->swap();
			setPressureBoundaryConditions();
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
			dyeAmount->src[dyeAmount->getPosition(i, 0)]   = (dyeAmount->src[dyeAmount->getPosition(i,   1)]);
			dyeAmount->src[dyeAmount->getPosition(i, h-1)] = (dyeAmount->src[dyeAmount->getPosition(i, h-2)]);
		}

		for(int j = 1; j < h - 1; j++)
		{
			u->src[u->getPosition(0, j)]   = -(u->src[u->getPosition(1,   j)]);
			u->src[u->getPosition(w-1, j)] = -(u->src[u->getPosition(w-2, j)]);
			v->src[v->getPosition(0, j)]   =  (v->src[v->getPosition(1,   j)]);
			v->src[v->getPosition(w-1, j)] =  (v->src[v->getPosition(w-2, j)]);
			dyeAmount->src[dyeAmount->getPosition(0, j)]   = (dyeAmount->src[dyeAmount->getPosition(1,   j)]);
			dyeAmount->src[dyeAmount->getPosition(w-1, j)] = (dyeAmount->src[dyeAmount->getPosition(w-2, j)]);
		}

		setPressureBoundaryConditions();
	}

	//set pressure in boundary cells, for use in jacobi iterations
	void setPressureBoundaryConditions()
	{
		for(int i = 1; i < w - 1; i++)
		{
			pressure->src[pressure->getPosition(i, 0)]   = -pressure->src[pressure->getPosition(i,   1)];
			pressure->src[pressure->getPosition(i, h-1)] = -pressure->src[pressure->getPosition(i, h-2)];
		}

		for(int j = 1; j < h - 1; j++)
		{
			pressure->src[pressure->getPosition(0, j)]   = -pressure->src[pressure->getPosition(1,   j)];
			pressure->src[pressure->getPosition(w-1, j)] = -pressure->src[pressure->getPosition(w-2, j)];
		}
	}

	//update the fluid domain through one timestep
	void update()
	{
		u->addInflow(20, 22, 230, 250, 15); // in m/sec
		dyeAmount->addInflow(20, 22, 230, 250, 1);
		u->advect(u, v);
		v->advect(u, v);
		dyeAmount->advect(u, v);
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

int main()
{
	FluidSolver *solver = new FluidSolver(WIDTH, HEIGHT);
	char frameName[1024];

	for(int frame = 0; frame != NUM_FRAMES; frame++)
	{
		solver->update();
		if(frame % 2 == 0)
		{
			sprintf(frameName, "frames/frame-%03d.png", frame / 2);
		}
		pngwriter image(WIDTH - 2, HEIGHT - 2, 0.0, frameName);
		for(int i = 0; i < WIDTH - 2; i++)
		{
			for(int j = 0; j < HEIGHT - 2; j++)
			{
				float colorScale = solver->dyeAmount->dst[solver->dyeAmount->getPosition(i+1, j+1)];
				image.plot(i, j, (double)colorScale, (double)colorScale, (double)colorScale);

			}
		}
		image.close();
		solver->swapAll();
	}

    return 0;
}
