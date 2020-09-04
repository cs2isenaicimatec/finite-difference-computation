#include	"sftest.h"

MAIN()
{
	Sfio_t*	f;
	char	buf[8*1024];
	int	i;
	Sfoff_t	s;

	/* test file resizing */
#if _lib_ftruncate
	if(!(f = sfopen(NIL(Sfio_t*), tstfile(0), "w")) )
		terror("Can't open file %s", tstfile(0));

	for(i = 0; i < sizeof(buf); ++i)
		buf[i] = '1';

	for(i = 0; i < 1024; ++i)
		if(sfwrite(f, buf, sizeof(buf)) != sizeof(buf) )
			terror("Can't write data");

	if(sfclose(f) < 0)
		terror("Can't sync/close file");
	if(!(f = sfopen(NIL(Sfio_t*), tstfile(0), "r+")) )
		terror("Can't open file %s again", tstfile(0));
	if(sfsize(f) != (s = 1024*sizeof(buf)) )
		terror("Bad file size");

	if(sfresize(f, s + sizeof(buf)) < 0)
		terror("Can't resize file");
	if(sfsize(f) != s+sizeof(buf))
		terror("Bad file size");

	sfseek(f, s, 0);
	if(sfread(f,buf,sizeof(buf)) != sizeof(buf))
		terror("Can't read data");

	for(i = 0; i < sizeof(buf); ++i)
		if(buf[i] != 0)
			terror("Bad data");
	sfclose(f);

	if(!(f = sfopen(NIL(Sfio_t*), tstfile(0), "r+")) )
		terror("Can't open file %s again", tstfile(0));
	if(sfsize(f) != s+sizeof(buf))
		terror("Bad file size");

	if(sfresize(f, s) < 0)
		terror("Can't resize file");
	sfclose(f);

	if(!(f = sfopen(NIL(Sfio_t*), tstfile(0), "r+")) )
		terror("Can't open file %s again", tstfile(0));
	if(sfsize(f) != s)
		terror("Bad file size");
#endif

	/* test resizing string stream */
	if(!(f = sfopen(NIL(Sfio_t*),  NIL(char*), "rws")) )
		terror("Can't open string stream");

	for(i = 0; i < sizeof(buf); ++i)
		buf[i] = '1';

	for(i = 0; i < 1024; ++i)
		if(sfwrite(f, buf, sizeof(buf)) != sizeof(buf) )
			terror("Can't write data");

	if(sfsize(f) != (s = 1024*sizeof(buf)) )
		terror("Bad stream size");

	if(sfresize(f, s + sizeof(buf)) < 0)
		terror("Can't resize stream");
	if(sfsize(f) != s+sizeof(buf))
		terror("Bad stream size");

	if(sfseek(f, s, 0) != s)
		terror("seek failed");
	if(sfread(f,buf,sizeof(buf)) != sizeof(buf))
		terror("Can't read data");

	for(i = 0; i < sizeof(buf); ++i)
		if(buf[i] != 0)
			terror("Bad data");

	if(sfresize(f, s) < 0)
		terror("Can't resize stream");
	if(sfsize(f) != s)
		terror("Bad stream size");

	TSTEXIT(0);
}
