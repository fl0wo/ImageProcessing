/**
 Created by Sabani Florian on 23/03/20.
*/
#include <stdio.h>
#include "ip_lib.h"
#include "bmp.h"
#include <sys/time.h>

// max and min macros
#define max2(x, y) (((x) > (y)) ? (x) : (y))
#define min2(x, y) (((x) < (y)) ? (x) : (y))

// set it 1 if want to use fast blur insted classic convolve gauss
#define USE_GAUSS_FASTBLUR_INSTEAD 0
#define FAST_BLUR_ID -49

// triple for that iterates a ip_mat "t"
// i for rows, j for columns, k for channels
#define ipmatloop(t,i,j,_k) for(int _k=0;_k<(t->k);_k++)for(int i=0;i<(t->h);i++)for(int j=0;j<(t->w);j++)

// macros for timer clock
#define START_TIMER struct timeval stop, start;gettimeofday(&start, NULL);
#define END_TIMER(mat) gettimeofday(&stop, NULL); long udelay = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec; printf("[debug] convultion filter took %ld ms on %dx%d image resolution\n",udelay/1000,mat->h,mat->w); 

int ip_mat_check_dims(ip_mat * a, ip_mat * b){return (a->h)==(b->h) && (a->w)==(b->w) && (a->k)==(b->k);}

/**
 * Print bitmap matrix rappresentation.
*/
void ip_mat_show(ip_mat * t){
    unsigned int i,l,j;
    printf("Matrix of size %d x %d x %d (hxwxk)\n",t->h,t->w,t->k);
    for (l = 0; l < t->k; l++) {
        printf("Slice %d\n", l);
        for(i=0;i<t->h;i++) {
            for (j = 0; j < t->w; j++) {
                printf("%f ", get_val(t,i,j,l));
            }
            printf("\n");
        }
        printf("\n");
    }
}

/**
 * Print stats of gien ip_mat
*/
void ip_mat_show_stats(ip_mat * t){
    unsigned int k;
    compute_stats(t);
    for(k=0;k<t->k;k++){
        printf("Channel %d:\n", k);
        printf("\t Min: %f\n", t->stat[k].min);
        printf("\t Max: %f\n", t->stat[k].max);
        printf("\t Mean: %f\n", t->stat[k].mean);
    }
}

/**
 * If error occurs, exit program.
*/
void onError(char* method){
    printf("[error] : %s", method);
    exit(1);
}

/**
 * Converts a given bitmap to a ip_mat struct.
*/
ip_mat * bitmap_to_ip_mat(Bitmap * img){
    unsigned int i=0,j=0;
    unsigned char R,G,B;
    unsigned int h = img->h;
    unsigned int w = img->w;
    ip_mat * out = ip_mat_create(h, w,3,0);
    for (i = 0; i < h; i++){
        for (j = 0; j < w; j++){
            bm_get_pixel(img, j,i,&R, &G, &B);
            set_val(out,i,j,0,(float) R);
            set_val(out,i,j,1,(float) G);
            set_val(out,i,j,2,(float) B);
        }
    }
    return out;
}

/**
 * Converts a ip_mat to a Bitmap.
*/
Bitmap * ip_mat_to_bitmap(ip_mat * t){
    Bitmap *b = bm_create(t->w,t->h);
    unsigned int i, j;
    for (i = 0; i < t->h; i++){
        for (j = 0; j < t->w; j++){
            bm_set_pixel(b, j,i, (unsigned char) get_val(t,i,j,0),
                    (unsigned char) get_val(t,i,j,1),
                    (unsigned char) get_val(t,i,j,2));
        }
    }
    return b;
}

/**
 * Return value cell of a ip_mat (i-row j-column k-channel)
 * j>=0 and k>=0 and i>=0 is non sense
*/
float get_val(ip_mat * a, unsigned int i,unsigned int j,unsigned int k){
    if(i<a->h && j<a->w &&k<a->k) return a->data[i][j][k];
    else{
        printf("uscito da %dx%d con %d : %d\n",a->h,a->w,i,j);
        onError("get_val");
    }
    return -1.0;
}
/**
 * Sets value to a cell of an ip_mat (i-row j-column k-channel)
 * j>=0 and k>=0 and i>=0 is non sense
*/
void set_val(ip_mat * a, unsigned int i,unsigned int j,unsigned int k, float v){
    if(i<a->h && j<a->w &&k<a->k) a->data[i][j][k]=v;
    else {
        printf("uscito da %dx%d con %d : %d\n",a->h,a->w,i,j);
        onError("set_val");
    }
}

/**
 * Returns normal random number generated.
*/
float get_normal_random(){
    float y1 = ( (float)(rand()) + 1. )/( (float)(RAND_MAX) + 1. );
    float y2 = ( (float)(rand()) + 1. )/( (float)(RAND_MAX) + 1. );
    return cos(2*PI*y2)*sqrt(-2.*log(y1));
}

/**
 * Creates and returns an ip_mat.
 * h : number of rows
 * w : number of columns
 * k : number of channel (usually 3)
*/
ip_mat * ip_mat_create(unsigned int h, unsigned int w,unsigned int k, float v){

    ip_mat* ipmat = malloc(sizeof(ip_mat));
    ipmat->w=w;
    ipmat->h=h;
    ipmat->k=k;
    ipmat->stat = malloc(sizeof(stats) * k); // per ogni channel

    ipmat->data = malloc(sizeof(float**) * h);
    for(int i=0;i<h;i++){
        ipmat->data[i] = malloc(sizeof(float*) * w);
        for(int j=0;j<w;j++){
            ipmat->data[i][j] = malloc(sizeof(float) * k);
            for(int _k=0;_k<k;_k++)
                set_val(ipmat,i,j,_k,v);
        }
    }

    return ipmat;
}

/* Libera la memoria (data, stat e la struttura) */
void ip_mat_free(ip_mat *a){
    if(!a)return;
    free(a->stat);
    unsigned int i,j;
    for(i=0; i<a->h; i++){
        for(j=0; j<a->w; j++)
            free(a->data[i][j]);
        free(a->data[i]);
    }
    free(a->data);
    free(a);
}

/**
 * Computes and sets the stats for every channel of an ip_mat.
 * Computes mins maxs and avgs for every channel of an ip_mat.
*/
void compute_stats(ip_mat * t){

    float sums[3] = {0,0,0},
    mins[3] = {FLT_MAX,FLT_MAX,FLT_MAX},
    maxs[3] = {FLT_MIN,FLT_MIN,FLT_MIN};

    ipmatloop(t,a,b,c){
        mins[c] = min2(mins[c], get_val(t,a,b,c));
        maxs[c] = max2(maxs[c], get_val(t,a,b,c));
        sums[c] += get_val(t,a,b,c);
    }

    for(int k=0;k<t->k;k++) {
        (t->stat)[k].min = mins[k];
        (t->stat[k]).max = maxs[k];
        (t->stat[k]).mean = sums[k] / (t->h * t->w);
    }
}

/* Inizializza una ip_mat con dimensioni w h e k.
 * Ogni elemento è generato da una gaussiana con media mean e varianza var */
/**
 * Fill ip_mat channels with random generated values.
 * Values are generated by gaussian function.
 * mean = mean of the gaussian function.
 * var = variance of the gaussian function.
 * Actually search about box-muller to understand more.
 * https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
 */    
// variance = standard deviation^2
void ip_mat_init_random(ip_mat * t, float mean, float var){// box-muller implementation
    ipmatloop(t,i,j,k)
        set_val(t,i,j,k, get_normal_random() * sqrt(var) + mean);
}

/**
 * Return a copy of an ip_mat.
 */
ip_mat * ip_mat_copy(ip_mat * t){
    ip_mat * clone = ip_mat_create(t->h,t->w,t->k,0.0F);

    ipmatloop(t,i,j,k){
        set_val(clone,i,j,k, get_val(t,i,j,k));
    }

    for(int k=0;k<t->k;k++) (clone->stat)[k] = (t->stat)[k];
    return clone;
}

/**
 * Return subset of the matrix.
 * t->data[row_start...row_end][col_start...col_end][0...k] (taking all channels)
 */
ip_mat * ip_mat_subset(ip_mat * t, unsigned int row_start, unsigned int row_end, unsigned int col_start, unsigned int col_end){
    int sub_width = col_end - col_start;
    int sub_height = row_end - row_start;

    if(sub_width<0 || sub_height<0) onError("ip_mat_subset parameters!");

    ip_mat * clone = ip_mat_create(sub_height,sub_width,t->k,0.0F);

    ipmatloop(clone,i,j,k){
        set_val(clone,i,j,k, get_val(t,i+row_start,j+col_start,k));
    }
    
    //compute_stats(clone);
    return clone;
}

/* Concats two ip_mat on a given dimension.
 * For example:
 * ip_mat_concat(ip_mat * a, ip_mat * b, 0);
 *      will create a new ip_mat of dimensions:
 *      out.h = a.h + b.h
 *      out.w = a.w = b.w
 *      out.k = a.k = b.k
 *
 * ip_mat_concat(ip_mat * a, ip_mat * b, 1);
 *      will create a new ip_mat of dimensions:
 *      out.h = a.h = b.h
 *      out.w = a.w + b.w
 *      out.k = a.k = b.k
 *
 * ip_mat_concat(ip_mat * a, ip_mat * b, 2);
 *      will create a new ip_mat of dimensions:
 *      out.h = a.h = b.h
 *      out.w = a.w = b.w
 *      out.k = a.k + b.k
 * */    
// se dim == 2 concateno sotto le righe
// se dim == 1 concateno a dx delle colonne
// se dim == 3 (sovrappongo le immagini con i canali)
ip_mat * ip_mat_concat(ip_mat * a, ip_mat * b, int dim){
    int dim_orders[3] = {2,1,3};
    int new_h = dim==dim_orders[0] ? a->h + b->h : a->h; // solo se 0 -> b under a
    int new_w = dim==dim_orders[1] ? a->w + b->w : a->w; // solo se 1 -> b alla dx di a
    int new_k = dim==dim_orders[2] ? a->k + b->k : a->k; // solo se 3 -> canals a + canals b
    // correct size
    ip_mat * clone = ip_mat_create(new_h,new_w,new_k,0.0F);
    // fill
    ipmatloop(clone,i,j,k){
        if((k>=a->k) || (j>=a->w) || (i>=a->h))       // se ho ho sforato a
            set_val(clone,i,j,k, get_val(b,             // vado in b
                i - (a->h) * (dim==dim_orders[0]),      // se dim è quella delle righe
                j - (a->w) * (dim==dim_orders[1]),      // se quella delle colonne
                k - (a->k) * (dim==dim_orders[2])       // se quella dei canali
            ));     
        else set_val(clone,i,j,k,get_val(a,i,j,k));
    }

    return clone;
}

/**** PARTE 1: OPERAZIONI MATEMATICHE FRA IP_MAT ****/

/** Performs sum of two ip_mat. */
ip_mat * ip_mat_sum(ip_mat * a, ip_mat * b){
    if(a->w!=b->w || a->h!=b->h || a->k!=b->k) onError("ip_mat_sum params");

    ip_mat * clone = ip_mat_create(a->h,a->w,a->k,0.0F);

    ipmatloop(clone,i,j,k)
        set_val(clone,i,j,k, get_val(a,i,j,k) + get_val(b,i,j,k));

    return clone;
}

/** Performs subs of two ip_mat. */
ip_mat * ip_mat_sub(ip_mat * a, ip_mat * b){
    if(a->w!=b->w || a->h!=b->h || a->k!=b->k) onError("ip_mat_sum params");

    ip_mat * clone = ip_mat_create(a->h,a->w,a->k,0.0F);

    ipmatloop(clone,i,j,k)
        set_val(clone,i,j,k, get_val(a,i,j,k) - get_val(b,i,j,k));

    return clone;
}

/** Performs a[i][j][k]*scalar on each cell of ip_mat a. */
ip_mat * ip_mat_mul_scalar(ip_mat *a, float scalar){
    ip_mat * clone = ip_mat_create(a->h,a->w,a->k,0.0F);

    ipmatloop(clone,i,j,k)
        set_val(clone,i,j,k, get_val(a,i,j,k) * scalar);

    return clone;
}

/** Performs a[i][j][k]+scalar on each cell of ip_mat a. */
ip_mat *  ip_mat_add_scalar(ip_mat *a, float scalar){
    ip_mat * clone = ip_mat_create(a->h,a->w,a->k,0.0F);
    ipmatloop(clone,i,j,k)
        set_val(clone,i,j,k, get_val(a,i,j,k) + scalar);
    return clone;
}

/* Calcola la media di due ip_mat a e b e la restituisce in output.*/
ip_mat * ip_mat_mean(ip_mat * a, ip_mat * b){
    ip_mat * clone = ip_mat_create(a->h,a->w,a->k,0.0F);
    ipmatloop(clone,i,j,k)
        set_val(clone,i,j,k, (get_val(a,i,j,k) + get_val(b,i,j,k)) / 2.0F );
    return clone;
}

/**** PARTE 2: SEMPLICI OPERAZIONI SU IMMAGINI ****/


/* Converte un'immagine RGB ad una immagine a scala di grigio.
 * Quest'operazione viene fatta calcolando la media per ogni pixel sui 3 canali
 * e creando una nuova immagine avente per valore di un pixel su ogni canale la media appena calcolata.
 * Avremo quindi che tutti i canali saranno uguali.
 * */
ip_mat * ip_mat_to_gray_scale(ip_mat * a){
    ip_mat * clone = ip_mat_create(a->h,a->w,a->k,0.0F);
    for(int i=0;i<a->h;i++)
        for(int j=0;j<a->w;j++) {
            float avg = 0;
            for (int k = 0; k < a->k; k++) avg += get_val(a,i,j,k);
            avg/=(a->k);
            for (int k = 0; k < a->k; k++) set_val(clone,i,j,k,avg);
        }
    return clone;
}

/* Effettua la fusione (combinazione convessa) di due immagini */
ip_mat * ip_mat_blend(ip_mat * a, ip_mat * b, float alpha){
    if(!ip_mat_check_dims(a,b)) onError("ip_mat_blend not same dimension");
    printf("same dim : %d", ip_mat_check_dims(a,b));
    ip_mat * clone = ip_mat_create(a->h,a->w,a->k,0.0F);
    ipmatloop(clone,i,j,k)
        set_val(clone,i,j,k,(alpha * get_val(a,i,j,k) + (1-alpha) * get_val(b,i,j,k)));
    return clone;
}

/* Operazione di brightening: aumenta la luminosità dell'immagine
 * aggiunge ad ogni pixel un certo valore*/
ip_mat * ip_mat_brighten(ip_mat * a, float bright){
    return ip_mat_add_scalar(a,bright);
}

/* Operazione di corruzione con rumore gaussiano:
 * Aggiunge del rumore gaussiano all'immagine, il rumore viene enfatizzato
 * per mezzo della variabile amount.
 * out = a + gauss_noise*amount
 * */
ip_mat * ip_mat_corrupt(ip_mat * a, float amount){
    amount/=255.0F;
    ip_mat * cloneA = ip_mat_copy(a);
    ip_mat * randomBtm = ip_mat_create(a->h,a->w,a->k,0.0F);
    ip_mat_init_random(randomBtm,0,255*255);
    ipmatloop(cloneA,i,j,k)
        set_val(cloneA,i,j,k, get_val(cloneA,i,j,k) +  get_val(randomBtm,i,j,k) * amount);
    return cloneA;
/*
    Alternativa, prendere il bmp a, generare random bmp b,
    fare blend di a con b in funzione di amount.

    ip_mat * cloneA = ip_mat_copy(a);
    ip_mat * randomBtm = ip_mat_create(a->h,a->w,a->k,0.0F);
    ip_mat_init_random(randomBtm,0,255*255);
    float realAmount = amount/255;
    printf("blendo con amoutn %f",amount);
    ip_mat * blended = ip_mat_blend(randomBtm,cloneA,realAmount);
    return blended;
*/
}

/**** PARTE 3: CONVOLUZIONE E FILTRI *****/

/* Aggiunge un padding all'immagine. Il padding verticale è pad_h mentre quello
 * orizzontale è pad_w.
 * L'output sarà un'immagine di dimensioni:
 *      out.h = a.h + 2*pad_h;
 *      out.w = a.w + 2*pad_w;
 *      out.k = a.k
 * con valori nulli sui bordi corrispondenti al padding e l'immagine "a" riportata
 * nel centro
 * */
ip_mat * ip_mat_padding(ip_mat * a, int pad_h, int pad_w){
    ip_mat * clone = ip_mat_create(a->h + pad_h*2,a->w + 2*pad_w,a->k,0.0F);
    for(int i=pad_h;i<a->h + pad_h;i++)
        for(int j=pad_w;j<a->w + pad_w;j++)
            for(int k=0;k<a->k;k++)
                set_val(clone,i-1,j-1,k,get_val(a,i-pad_h,j-pad_w,k));
    return clone;
}

/**
 * Per puro divertimento è stato creato un algoritmo alternativo al gauss per l'effetto "sfuocato"
 * 
 */

ip_mat * fastBoxBlur(ip_mat * img, int radius) {

    printf("raggiuooo = %d \n",radius);

    if (radius%2==0)radius++;
    ip_mat * hor_blur = ip_mat_copy(img);
    float Avg =(float)1/radius;

    for (int j=0;j<img->h;j++) {
        float h_sum[3] = {0.0f,0.0f,0.0f};
        float i_avg[3] = {0.0f,0.0f,0.0f};

        for (int x=0;x<radius;x++)
            for(int k=0;k<img->k;k++)
                h_sum[k] += get_val(img,j,x,k);

        for(int k=0;k<img->k;k++)
            i_avg[k] = h_sum[k] * Avg;

        for (int i=0;i<img->w;i++) {
            if((i+1+radius/2<img->w && i-radius/2 >= 0) && j>=0 && j<img->h){
                for(int k=0;k<img->k;k++)
                    h_sum[k] -= get_val(img,j,i-radius/2,k);
                for(int k=0;k<img->k;k++)
                    h_sum[k] += get_val(img,j,i+1 + radius/2,k);

                for(int k=0;k<img->k;k++)
                    i_avg[k] = h_sum[k] * Avg;
            }

            for(int k=0;k<img->k;k++)
                set_val(hor_blur,j,i,k,i_avg[k]);
        }
    }

    ip_mat * total = ip_mat_copy(hor_blur);

    for (int i=0;i<total->w;i++) {
        float t_sum[3] = {0.0f,0.0f,0.0f};
        float i_avg[3] = {0.0f,0.0f,0.0f};

        for (int y=0;y<radius;y++)
            for(int k=0;k<total->k;k++)
                t_sum[k] += get_val(total,y,i,k);

        for(int k=0;k<total->k;k++)
            i_avg[k] = t_sum[k] * Avg;

        for (int j=0;j<hor_blur->h;j++) {
            if(  (j-radius/2)>=0 && (j+1+radius/2)<hor_blur->h && i<hor_blur->w){
                for(int k=0;k<hor_blur->k;k++)
                    t_sum[k] -= get_val(hor_blur,j-radius/2,i,k);

                for(int k=0;k<hor_blur->k;k++)
                    t_sum[k] += get_val(hor_blur,j+1+radius/2,i,k);

                for(int k=0;k<hor_blur->k;k++)
                    i_avg[k] = t_sum[k] * Avg;
            }
            for(int k=0;k<hor_blur->k;k++)
                set_val(total,j,i,k,i_avg[k]);
        }
    }
    return total;
}

int boxesForGaussian(double sigma, int n,int sizes) {
    double wIdeal = sqrt((12 * sigma * sigma / n) + 1);
    int wl = floor(wIdeal);
    if (wl%2 == 0) wl--;
    double wu = wl + 2;
    double mIdeal=(12*sigma*sigma-n*wl*wl-4*n*wl-3*n)/(-4*wl-4);
    int m = round(mIdeal);
    return (0 < m) ? wl : wu;
    /*
    for multiple sizes
    for (int i = 0; i < n; i++) {
        if (i < m) sizes[i] = (int) wl;
        else sizes[i] = (int) wu;
    }
    */
}

ip_mat * fastGaussianBlur(ip_mat * src, int radius) {
    int gaussianBox = boxesForGaussian(radius,3, 0);
    ip_mat * img = fastBoxBlur(src, gaussianBox);
    return img;
}

/* Effettua la convoluzione di un ip_mat "a" con un ip_mat "f".
 * La funzione restituisce un ip_mat delle stesse dimensioni di "a".
 * Per il filtro di gauss si può decidere di usare un algoritmo di fast-blur
 * alternativo alla classica convoluzione.
 * Estremamente più veloce.
 * */
ip_mat * ip_mat_convolve(ip_mat * a, ip_mat * f){
    START_TIMER

    if(USE_GAUSS_FASTBLUR_INSTEAD && get_val(f,0,0,0) == FAST_BLUR_ID){
        ip_mat* fastBlur = fastGaussianBlur(a,7);
        END_TIMER(fastBlur)
        return fastBlur;
    }

    printf("f : %d , %d , %d\n",f->h,f->w,f->k);

    ip_mat* with_pad = ip_mat_padding(a, (f->h-1)/2 + 1, (f->w-1)/2 + 1);
    ip_mat* answ = ip_mat_create(a->h,a->w,a->k,0.0F);

    printf("with_pad : %d , %d , %d\n",with_pad->h,with_pad->w,with_pad->k);
    printf("answ : %d , %d , %d\n",answ->h,answ->w,answ->k);

    ipmatloop(answ,i,j,k){
        float media = 0;
        for(int gap_i=0;gap_i<f->h;gap_i++)
            for(int gap_j=0;gap_j<f->w;gap_j++)
                media += get_val(with_pad,gap_i + i,gap_j + j,k) * get_val(f,gap_i,gap_j,0);
        set_val(answ,i,j,k,media);
    }

    END_TIMER(answ)

    return answ;
}
/*Converts matrix to ip_mat*/
ip_mat * to_ip_mat(float *mat,int rows,int cols,int canals) {
    ip_mat *clone = ip_mat_create(rows, cols, 1, 0.0F);
    ipmatloop(clone,i,j,k)
        set_val(clone,i,j,k,(*((mat+i*cols) + j)));
    return clone;
}
/* Crea un filtro di sharpening */
ip_mat * create_sharpen_filter(){
    const float mat[3][3] = {{0,-1,0},{-1,5,-1},{0,-1,0}};
    return to_ip_mat((float *)mat,3,3,1);
}

/* Crea un filtro per rilevare i bordi */
ip_mat * create_edge_filter(){
    const float mat[3][3] = {{-1,-1,-1},{-1,8,-1},{-1,-1,-1}};
    return to_ip_mat((float *)mat,3,3,1);
}

/* Crea un filtro per aggiungere profondità */
ip_mat * create_emboss_filter(){
    const float mat[3][3] = {{-2,-1,0},{-1,1,1},{0,1,2}};
    return to_ip_mat((float *)mat,3,3,1);
}

/* Crea un filtro medio per la rimozione del rumore */
ip_mat * create_average_filter(unsigned int w,unsigned int h,unsigned int k){
    float mat[h][w];float c=1.0/(w*h);for(int i=0;i<h;i++)for(int j=0;j<w;j++) mat[i][j]=c;
    return to_ip_mat((float *)mat,h,w,k);
}

/* Crea un filtro gaussiano per la rimozione del rumore */
ip_mat * create_gaussian_filter(unsigned int wu,unsigned int hu,unsigned int ku, float sigma){ // 1
    int w=wu,h=hu,k=ku;
    if(USE_GAUSS_FASTBLUR_INSTEAD){
        const float mat2[3][3] = {{FAST_BLUR_ID,-1,0},{-1,1,1},{0,1,2}};
        return to_ip_mat((float *)mat2,3,3,1);
    }
    w+=w%2!=0;
    h+=h%2!=0;
    float mat[h][w];double r,s=2.0*sigma*sigma;double sum=0.0;
    for(int x=-(h/2);x<=h/2;x++) {
        for(int y=-(w/2);y<=w/2;y++) {
            r=sqrt(x*x+y*y);
            mat[x+(h/2)][y+(w/2)]=(exp(-(r*r)/s))/(PI*s);
            //printf("%f\n",mat[x+(h/2)][y+(w/2)]);
            sum+=mat[x+(h/2)][y+(w/2)];
        }
    }
    //printf("sum : %f\n",sum);
    // normalizza
    for (int i=0;i<h;i++){
        for (int j=0;j<w;j++){
            mat[i][j] /= sum;
            //printf("%f ",mat[i][j]);
        }
        //printf("\n");
    }

    return to_ip_mat((float *)mat,h,w,k);
}


/* Effettua una riscalatura dei dati tale che i valori siano in [0,new_max].
 * Utilizzate il metodo compute_stat per ricavarvi il min, max per ogni canale.
 *
 * I valori sono scalati tramite la formula valore-min/(max - min)
 *
 * Si considera ogni indice della terza dimensione indipendente, quindi l'operazione
 * di scalatura va ripetuta per ogni "fetta" della matrice 3D.
 * Successivamente moltiplichiamo per new_max gli elementi della matrice in modo da ottenere un range
 * di valori in [0,new_max].
 * */
void rescale(ip_mat * t, float new_max){
    compute_stats(t);
    ipmatloop(t,i,j,k)
        set_val(t,i,j,k, ((get_val(t,i,j,k)-t->stat[k].min)/(t->stat[k].max-t->stat[k].min))  *new_max);
}

/* Nell'operazione di clamping i valori <low si convertono in low e i valori >high in high.*/
void clamp(ip_mat * t, float low, float high){
    ipmatloop(t,i,j,k){
        set_val(t,i,j,k,max2(low,get_val(t,i,j,k)));
        set_val(t,i,j,k,min2(high,get_val(t,i,j,k)));
    }
}