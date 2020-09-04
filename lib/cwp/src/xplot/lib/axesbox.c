/* Copyright (c) Colorado School of Mines, 2011.*/
/* All rights reserved.                       */

/* Copyright (c) Colorado School of Mines, 2011.*/
/* All rights reserved.                       */

/* AXESBOX: $Revision: 1.13 $ ; $Date: 2011/11/21 17:02:44 $	*/

/*********************** self documentation **********************/
/*****************************************************************************
AXESBOX - Functions to draw axes in X-windows graphics

xDrawAxesBox	draw a labeled axes box
xSizeAxesBox	determine optimal origin and size for a labeled axes box

*****************************************************************************
Function Prototypes:
void xDrawAxesBox (Display *dpy, Window win,
	int x, int y, int width, int height,
	float x1beg, float x1end, float p1beg, float p1end,
	float d1num, float f1num, int n1tic, int grid1, char *label1,
	float x2beg, float x2end, float p2beg, float p2end,
	float d2num, float f2num, int n2tic, int grid2, char *label2,
	char *labelfont, char *title, char *titlefont, 
	char *axescolor, char *titlecolor, char *gridcolor,
	int style);
void xSizeAxesBox (Display *dpy, Window win, 
	char *labelfont, char *titlefont, int style,
	int *x, int *y, int *width, int *height);

*****************************************************************************
xDrawAxesBox:
Input:
dpy		display pointer
win		window
x		x coordinate of upper left corner of box
y		y coordinate of upper left corner of box
width		width of box
height		height of box
x1beg		axis value at beginning of axis 1
x1end		axis value at end of axis 1
p1beg		pad value at beginning of axis 1
p1end		pad value at end of axis 1
d1num		numbered tic increment for axis 1 (0.0 for automatic)
f1num		first numbered tic for axis 1
n1tic		number of tics per numbered tic for axis 1
grid1		grid code for axis 1:  NONE, DOT, DASH, or SOLID
label1		label for axis 1
x2beg		axis value at beginning of axis 2
x2end		axis value at end of axis 2
p2beg		pad value at beginning of axis 2
p2end		pad value at end of axis 2
d2num		numbered tic increment for axis 2 (0.0 for automatic)
f2num		first numbered tic for axis 2
n2tic		number of tics per numbered tic for axis 2
grid2		grid code for axis 2:  NONE, DOT, DASH, or SOLID
label2		label for axis 2
labelfont	name of font to use for axes labels
title		axes box title
titlefont	name of font to use for title
axescolor	name of color to use for axes
titlecolor	name of color to use for title
gridcolor	name of color to use for grid
int style	NORMAL (axis 1 on bottom, axis 2 on left)
		SEISMIC (axis 1 on left, axis 2 on top)

******************************************************************************
xSizeAxesBox:
Input:
dpy		display pointer
win		window
labelfont	name of font to use for axes labels
titlefont	name of font to use for title
int style	NORMAL (axis 1 on bottom, axis 2 on left)
		SEISMIC (axis 1 on left, axis 2 on top)

Output:
x		x coordinate of upper left corner of box
y		y coordinate of upper left corner of box
width		width of box
height		height of box
******************************************************************************
{
	XFontStruct *fa,*ft;
******************************************************************************
Notes:
xDrawAxesBox:
will determine the numbered tic incremenet and first
numbered tic automatically, if the specified increment is zero.

Pad values must be specified in the same units as the corresponding
axes values.  These pads are useful when the contents of the axes box
requires more space than implied by the axes values.  For example,
the first and last seismic wiggle traces plotted inside an axes box
will typically extend beyond the axes values corresponding to the
first and last traces.  However, all tics will lie within the limits
specified in the axes values (x1beg, x1end, x2beg, x2end).

xSizeAxesBox:
is intended to be used prior to xDrawAxesBox.

An "optimal" axes box is one that more or less fills the window, 
with little wasted space around the edges of the window.

******************************************************************************
Author:		Dave Hale, Colorado School of Mines, 01/27/90
*****************************************************************************/
/**************** end self doc ********************************/

#include "xplot.h"

void
xDrawAxesBox (Display *dpy, Window win,
	int x, int y, int width, int height,
	float x1beg, float x1end, float p1beg, float p1end,
	float d1num, float f1num, int n1tic, int grid1, char *label1,
	float x2beg, float x2end, float p2beg, float p2end,
	float d2num, float f2num, int n2tic, int grid2, char *label2,
	char *labelfont, char *title, char *titlefont, 
	char *axescolor, char *titlecolor, char *gridcolor,
	int style)
/*****************************************************************************
draw a labeled axes box
******************************************************************************
Input:
dpy		display pointer
win		window
x		x coordinate of upper left corner of box
y		y coordinate of upper left corner of box
width		width of box
height		height of box
x1beg		axis value at beginning of axis 1
x1end		axis value at end of axis 1
p1beg		pad value at beginning of axis 1
p1end		pad value at end of axis 1
d1num		numbered tic increment for axis 1 (0.0 for automatic)
f1num		first numbered tic for axis 1
n1tic		number of tics per numbered tic for axis 1
grid1		grid code for axis 1:  NONE, DOT, DASH, or SOLID
label1		label for axis 1
x2beg		axis value at beginning of axis 2
x2end		axis value at end of axis 2
p2beg		pad value at beginning of axis 2
p2end		pad value at end of axis 2
d2num		numbered tic increment for axis 2 (0.0 for automatic)
f2num		first numbered tic for axis 2
n2tic		number of tics per numbered tic for axis 2
grid2		grid code for axis 2:  NONE, DOT, DASH, or SOLID
label2		label for axis 2
labelfont	name of font to use for axes labels
title		axes box title
titlefont	name of font to use for title
axescolor	name of color to use for axes
titlecolor	name of color to use for title
gridcolor	name of color to use for grid
int style	NORMAL (axis 1 on bottom, axis 2 on left)
		SEISMIC (axis 1 on left, axis 2 on top)
******************************************************************************
Notes:
xDrawAxesBox will determine the numbered tic incremenet and first
numbered tic automatically, if the specified increment is zero.

Pad values must be specified in the same units as the corresponding
axes values.  These pads are useful when the contents of the axes box
requires more space than implied by the axes values.  For example,
the first and last seismic wiggle traces plotted inside an axes box
will typically extend beyond the axes values corresponding to the
first and last traces.  However, all tics will lie within the limits
specified in the axes values (x1beg, x1end, x2beg, x2end).
******************************************************************************
Author:		Dave Hale, Colorado School of Mines, 01/27/90
*****************************************************************************/
{
	GC gca,gct,gcg;
	XGCValues *values=NULL;
	XColor scolor,ecolor;
	XFontStruct *fa,*ft;
	XWindowAttributes wa;
	Colormap cmap;
	int labelca,labelcd,labelch,labelcw,titleca,
		ntic,xa,ya,tw,ticsize,ticb,numb,labelb,lstr,grided,grid,
		n1num,n2num;
	float dnum,fnum,dtic,amin,amax,base,scale,anum,atic,azero;
	char str[256],dash[2],*label;

	/* create graphics contexts */
	gca = XCreateGC(dpy,win,0,values);
	gct = XCreateGC(dpy,win,0,values);
	gcg = XCreateGC(dpy,win,0,values);

	/* get and set fonts and determine character dimensions */
	fa = XLoadQueryFont(dpy,labelfont);
	if (fa==NULL) fa = XLoadQueryFont(dpy,"fixed");
	if (fa==NULL) {
		fprintf(stderr,"Cannot load/query labelfont=%s\n",labelfont);
		exit(-1);
	}
	XSetFont(dpy,gca,fa->fid);
	labelca = fa->max_bounds.ascent;
	labelcd = fa->max_bounds.descent;
	labelch = fa->max_bounds.ascent+fa->max_bounds.descent;
	labelcw = fa->max_bounds.lbearing+fa->max_bounds.rbearing;
	ft = XLoadQueryFont(dpy,titlefont);
	if (ft==NULL) ft = XLoadQueryFont(dpy,"fixed");
	if (ft==NULL) {
		fprintf(stderr,"Cannot load/query titlefont=%s\n",titlefont);
		exit(-1);
	}
	XSetFont(dpy,gct,ft->fid);
	titleca = ft->max_bounds.ascent;

	/* determine window's current colormap */
	XGetWindowAttributes(dpy,win,&wa);
	cmap = wa.colormap;

	/* get and set colors */
	if (XAllocNamedColor(dpy,cmap,axescolor,&scolor,&ecolor))
		XSetForeground(dpy,gca,ecolor.pixel);
	else
		XSetForeground(dpy,gca,1L);
	if (XAllocNamedColor(dpy,cmap,titlecolor,&scolor,&ecolor))
		XSetForeground(dpy,gct,ecolor.pixel);
	else
		XSetForeground(dpy,gct,1L);
	if (XAllocNamedColor(dpy,cmap,gridcolor,&scolor,&ecolor))
		XSetForeground(dpy,gcg,ecolor.pixel);
	else
		XSetForeground(dpy,gcg,1L);

	/* determine tic size */
	ticsize = labelcw;

	/* determine numbered tic intervals */
	if (d1num==0.0) {
		n1num = (style==NORMAL ? width : height)/(8*labelcw);
		scaxis(x1beg,x1end,&n1num,&d1num,&f1num);
	}
	if (d2num==0.0) {
		n2num = (style==NORMAL ? height : width)/(8*labelcw);
		scaxis(x2beg,x2end,&n2num,&d2num,&f2num);
	}

	/* draw horizontal axis */
	if (style==NORMAL) {
		amin = (x1beg<x1end)?x1beg:x1end;
		amax = (x1beg>x1end)?x1beg:x1end;
		dnum = d1num;  fnum = f1num;  ntic = n1tic;
		scale = width/(x1end+p1end-x1beg-p1beg);
		base = x-scale*(x1beg+p1beg);
		ya = y+height;
		ticb = ticsize;
		numb = ticb+labelca;
		labelb = numb+labelch;
		grid = grid1;
		label = label1;
	} else {
		amin = (x2beg<x2end)?x2beg:x2end;
		amax = (x2beg>x2end)?x2beg:x2end;
		dnum = d2num;  fnum = f2num;  ntic = n2tic;
		scale = width/(x2end+p2end-x2beg-p2beg);
		base = x-scale*(x2beg+p2beg);
		ya = y;
		ticb = -ticsize;
		numb = ticb-labelcd;
		labelb = numb-labelch;
		grid = grid2;
		label = label2;
	}
	if (grid==SOLID)
		grided = True;
	else if (grid==DASH) {
		grided = True;
		XSetLineAttributes(dpy,gcg,1L,LineOnOffDash,CapButt,JoinMiter);
		dash[0] = 8;  dash[1] = 4;
		XSetDashes(dpy,gcg,0,dash,2);
	} else if (grid==DOT) {
		grided = True;
		XSetLineAttributes(dpy,gcg,1L,LineOnOffDash,CapButt,JoinMiter);
		dash[0] = 1;  dash[1] = 4;
		XSetDashes(dpy,gcg,0,dash,2);
	} else
		grided = False;
	azero = 0.0001*(amax-amin);
	for (anum=fnum; anum<=amax; anum+=dnum) {
		if (anum<amin) continue;
		xa = base+scale*anum;
		if (grided) XDrawLine(dpy,win,gcg,xa,y,xa,y+height);
		XDrawLine(dpy,win,gca,xa,ya,xa,ya+ticb);
		if (anum>-azero && anum<azero)
			sprintf(str,"%1.5g",0.0);
		else
			sprintf(str,"%1.5g",anum);
		lstr = (int) strlen(str);
		tw = XTextWidth(fa,str,lstr);
		XDrawString(dpy,win,gca,xa-tw/2,ya+numb,str,lstr);
	}
	dtic = dnum/ntic;
	for (atic=fnum-ntic*dtic-dtic; atic<=amax; atic+=dtic) {
		if (atic<amin) continue;
		xa = base+scale*atic;
		XDrawLine(dpy,win,gca,xa,ya,xa,ya+ticb/2);
	}
	lstr = (int) strlen(label);
	tw = XTextWidth(fa,label,lstr);
	XDrawString(dpy,win,gca,x+width-tw,ya+labelb,label,lstr);

	/* draw vertical axis */
	if (style==NORMAL) {
		amin = (x2beg<x2end)?x2beg:x2end;
		amax = (x2beg>x2end)?x2beg:x2end;
		dnum = d2num;  fnum = f2num;  ntic = n2tic;
		scale = -height/(x2end+p2end-x2beg-p2beg);
		base = y+height-scale*(x2beg+p2beg);
		grid = grid2;
		label = label2;
	} else {
		amin = (x1beg<x1end)?x1beg:x1end;
		amax = (x1beg>x1end)?x1beg:x1end;
		dnum = d1num;  fnum = f1num;  ntic = n1tic;
		scale = height/(x1end+p1end-x1beg-p1beg);
		base = y-scale*(x1beg+p1beg);
		grid = grid1;
		label = label1;
	}
	xa = x;
	ticb = -ticsize;
	numb = ticb-ticsize/4;
	if (grid==SOLID)
		grided = True;
	else if (grid==DASH) {
		grided = True;
		XSetLineAttributes(dpy,gcg,1L,LineOnOffDash,CapButt,JoinMiter);
		dash[0] = 8;  dash[1] = 4;
		XSetDashes(dpy,gcg,0,dash,2);
	} else if (grid==DOT) {
		grided = True;
		XSetLineAttributes(dpy,gcg,1L,LineOnOffDash,CapButt,JoinMiter);
		dash[0] = 1;  dash[1] = 4;
		XSetDashes(dpy,gcg,0,dash,2);
	} else
		grided = False;
	azero = 0.0001*(amax-amin);
	for (anum=fnum; anum<=amax; anum+=dnum) {
		if (anum<amin) continue;
		ya = base+scale*anum;
		if (grided) XDrawLine(dpy,win,gcg,x,ya,x+width,ya);
		XDrawLine(dpy,win,gca,xa,ya,xa+ticb,ya);
		if (anum>-azero && anum<azero)
			sprintf(str,"%1.5g",0.0);
		else
			sprintf(str,"%1.5g",anum);
		lstr = (int) strlen(str);
		tw = XTextWidth(fa,str,lstr);
		XDrawString(dpy,win,gca,xa+numb-tw,ya+labelca/4,str,lstr);
	}
	dtic = dnum/ntic;
	for (atic=fnum-ntic*dtic-dtic; atic<=amax; atic+=dtic) {
		if (atic<amin) continue;
		ya = base+scale*atic;
		XDrawLine(dpy,win,gca,xa,ya,xa+ticb/2,ya);
	}
	lstr = (int) strlen(label);
	if (style==NORMAL)
		XDrawString(dpy,win,gca,
			x+ticb-9*labelcw,
			y+labelca/4-labelch,label,lstr);
	else
		XDrawString(dpy,win,gca,
			x+ticb-9*labelcw,
			y+height+labelca/4+labelch,label,lstr);
	
	/* draw title */
	lstr = (int) strlen(title);
	tw = XTextWidth(ft,title,lstr);
	if (style==NORMAL)
		XDrawString(dpy,win,gct,
			x+width/2-tw/2,
			y+labelca/4-labelch-labelch,title,lstr);
	else
		XDrawString(dpy,win,gct,
			x+width/2-tw/2,
			y+height+labelca/4+labelch+titleca,title,lstr);

	/* draw axes box */
	XDrawRectangle(dpy,win,gca,x,y,width,height);

	/* free resources before returning */
	XFreeGC(dpy,gca);
	XFreeGC(dpy,gct);
	XFreeGC(dpy,gcg);
	XFreeFont(dpy,fa);
	XFreeFont(dpy,ft);
}

void
xSizeAxesBox (Display *dpy, Window win, 
	char *labelfont, char *titlefont, int style,
	int *x, int *y, int *width, int *height)
/*****************************************************************************
determine optimal origin and size for a labeled axes box
******************************************************************************
Input:
dpy		display pointer
win		window
labelfont	name of font to use for axes labels
titlefont	name of font to use for title
int style	NORMAL (axis 1 on bottom, axis 2 on left)
		SEISMIC (axis 1 on left, axis 2 on top)

Output:
x		x coordinate of upper left corner of box
y		y coordinate of upper left corner of box
width		width of box
height		height of box
******************************************************************************
Notes:
xSizeAxesBox is intended to be used prior to xDrawAxesBox.

An "optimal" axes box is one that more or less fills the window, 
with little wasted space around the edges of the window.
******************************************************************************
Author:		Dave Hale, Colorado School of Mines, 01/27/90
*****************************************************************************/
{
	XFontStruct *fa,*ft;
	XWindowAttributes attr;
	int labelch,labelcw,titlech,bl,bt,br,bb;

	/* get fonts and determine character dimensions */
	fa = XLoadQueryFont(dpy,labelfont);
	if (fa==NULL) fa = XLoadQueryFont(dpy,"fixed");
	if (fa==NULL) {
		fprintf(stderr,"Cannot load/query labelfont=%s\n",labelfont);
		exit(-1);
	}
	labelch = fa->max_bounds.ascent+fa->max_bounds.descent;
	labelcw = fa->max_bounds.lbearing+fa->max_bounds.rbearing;
	ft = XLoadQueryFont(dpy,titlefont);
	if (ft==NULL) ft = XLoadQueryFont(dpy,"fixed");
	if (ft==NULL) {
		fprintf(stderr,"Cannot load/query titlefont=%s\n",titlefont);
		exit(-1);
	}
	titlech = ft->max_bounds.ascent+ft->max_bounds.descent;

	/* determine axes box origin and size */
	XGetWindowAttributes(dpy,win,&attr);
	bl = 10*labelcw;
	br = attr.width-5*labelcw;
	while (br<=bl) {
		br += labelcw;
		bl -= labelcw;
	}
	if (bl<0) bl = 0;
	if (br>attr.width) br = attr.width;
	if (style==NORMAL) {
		bt = labelch+labelch/2+titlech;
		bb = attr.height-3*labelch;
	} else {
		bt = 3*labelch;
		bb = attr.height-labelch-labelch/2-titlech;
	}
	while (bb<=bt) {
		bb += labelch;
		bt -= labelch;
	}
	if (bt<0) bt = 0;
	if (bb>attr.height) bb = attr.height;
	
	*x = bl;
	*y = bt;
	*width = br-bl;
	*height = bb-bt;

	XFreeFont(dpy,fa);
	XFreeFont(dpy,ft);
}
